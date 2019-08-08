#include <cmath>
// #include <execution>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include <stdtracer>

DEFINE_TRACE_CONTEXTS

template <typename T> std::vector<T> range(T begin, T end, T step)
{
    std::vector<T> lst;
    for (T x = begin; x < end; x += step) { lst.push_back(x); }
    return lst;
}

template <typename T> T binom_pmf(int n, T p, int i)
{
    return std::exp(std::lgamma(static_cast<T>(n + 1)) -
                    std::lgamma(static_cast<T>(i + 1)) -
                    std::lgamma(static_cast<T>(n - i + 1))  //
                    + i * std::log(p) + (n - i) * std::log(1 - p));
}

template <typename T> T poisson_pmf(T p, int k)
{
    return std::exp(k * std::log(p) - p - std::lgamma(static_cast<T>(k + 1)));
}

template <typename T> float thrshTransExtd(T nu, int k, T th)
{
    T s = 0;
    for (int i = static_cast<int>(th * k); i < k; ++i) {
        s += binom_pmf(k - 1, nu, i);
    }
    return s;
}

template <typename T> float H(T theta, T nu, T z)
{
    // TRACE_SCOPE(__func__);

    const int k_max = 1000;
    T h = 0;
    for (int k = 1; k <= k_max; ++k) {
        const T p_k = poisson_pmf(z, k);
        h += static_cast<T>(k) / z * p_k * thrshTransExtd(nu, k, theta);
    }
    return h;
}

template <typename T>
std::vector<T> nu_critical_1(const std::vector<T> &z_list,
                             const std::vector<T> &theta_list,
                             const std::vector<T> &x_list)
{
    TRACE_SCOPE(__func__);

    const int z_size = z_list.size();
    const int theta_size = theta_list.size();
    const int x_size = x_list.size();

    const auto idx = [=](int i, int j, int k) {
        // [z_size, theta_size, x_size]
        // [i     , j         ,      k]
        return (i * theta_size + j) * x_size + k;
    };

    std::vector<T> results(z_size * theta_size * x_size);
    std::cout << "result size: " << results.size() << std::endl;

    const int m = std::thread::hardware_concurrency();

    for (int i = 0; i < z_size; ++i) {
        // printf("%d/%d\n", i, z_size);
        {
            // TRACE_SCOPE("for z");
#pragma omp parallel for
            for (int j = 0; j < theta_size; ++j) {
                for (int k = 0; k < x_size; ++k) {
                    const T h = H(theta_list[j], x_list[k], z_list[i]);
                    results[idx(i, j, k)] = h;
                }
            }
        }
    }
    return results;
}

template <typename T>
std::vector<std::tuple<T, T, T>> cartesian_product(const std::vector<T> &x,
                                                   const std::vector<T> &y,
                                                   const std::vector<T> &z)
{
    std::vector<std::tuple<T, T, T>> p;
    for (auto a : x) {
        for (auto b : y) {
            for (auto c : z) { p.push_back(std::make_tuple(a, b, c)); }
        }
    }
    return p;
}

template <typename T, typename S, typename F>
void map(const std::vector<T> &x, std::vector<S> &y, const F &f)
{
    std::transform(x.begin(), x.end(), y.begin(), f);
}

int ceil_div(int a, int b) { return (a / b) + (a % b ? 1 : 0); }

template <typename T, typename S, typename F>
void pmap(int m, const std::vector<T> &x, std::vector<S> &y, const F &f)
{
    const int n = x.size();
    const int k = ceil_div(n, m);
    std::vector<std::thread> ths;
    for (int i = 0; i < n; i += k) {
        ths.push_back(std::thread([&] {
            const int j = std::min(i + k, n);
            std::transform(x.begin() + i, x.begin() + j, y.begin() + i, f);
        }));
    }
    for (auto &t : ths) { t.join(); }
}

template <typename T>
std::vector<T> nu_critical_2(const std::vector<T> &z_list,
                             const std::vector<T> &theta_list,
                             const std::vector<T> &x_list)
{
    TRACE_SCOPE(__func__);
    const auto grid = cartesian_product(z_list, theta_list, x_list);
    std::vector<T> results(grid.size());
    std::cout << "result size: " << results.size() << std::endl;

    const int m = std::thread::hardware_concurrency();

    pmap(m, grid, results, [](const std::tuple<T, T, T> &p) {
        const auto [z, theta, x] = p;
        return H(theta, x, z);
    });

    return results;
}

template <typename T>
void save_result(const std::vector<T> &z_list,      //
                 const std::vector<T> &theta_list,  //
                 const std::vector<T> &x_list,      //
                 const std::vector<T> &results)
{
    TRACE_SCOPE(__func__);
    FILE *fp = fopen("results.txt", "w");
    int i = 0;
    for (auto z : z_list) {
        for (auto th : theta_list) {
            for (auto x : x_list) {
                const T h = results[i++];
                fprintf(fp, "%f %f %f %f\n", th, x, z, h);
            }
        }
    }
    fclose(fp);
}

int main()
{
    // TRACE_SCOPE(__func__);

    using T = float;

    // const T z_step = 0.1;
    // const T theta_step = 0.01;
    // const T x_step = 0.002;

    const T z_step = 0.5;
    const T theta_step = 0.5;
    const T x_step = 0.1;

    const auto z_list = range<T>(1, 16, z_step);
    const auto theta_list = range<T>(0.01, 1, theta_step);
    const auto x_list = range<T>(0.001, 1, x_step);

    std::cout << "z list: " << z_list.size() << std::endl;
    std::cout << "theta list: " << theta_list.size() << std::endl;
    std::cout << "x list: " << x_list.size() << std::endl;

    {
        auto results = nu_critical_1(z_list, theta_list, x_list);
        save_result(z_list, theta_list, x_list, results);
    }
    {
        auto results = nu_critical_2(z_list, theta_list, x_list);
        save_result(z_list, theta_list, x_list, results);
    }
    return 0;
}
