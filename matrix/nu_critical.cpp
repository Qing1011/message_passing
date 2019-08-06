#include <cmath>
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
std::vector<T> nu_critical(const std::vector<T> &z_list,
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
        printf("%d/%d\n", i, z_size);
        {
            TRACE_SCOPE("for z");
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

int main()
{
    // TRACE_SCOPE(__func__);

    using T = float;

    // const T z_step = 0.1;
    // const T theta_step = 0.01;
    // const T x_step = 0.002;

    const T z_step = 0.1;
    const T theta_step = 0.01;
    const T x_step = 0.2;

    const auto z_list = range<T>(1, 16, z_step);
    const auto theta_list = range<T>(0.01, 1, theta_step);
    const auto x_list = range<T>(0.001, 1, x_step);

    std::cout << "z list: " << z_list.size() << std::endl;
    std::cout << "theta list: " << theta_list.size() << std::endl;
    std::cout << "x list: " << x_list.size() << std::endl;

    auto results = nu_critical(z_list, theta_list, x_list);

    // save result
    {
        FILE *fp = fopen("results.txt", "w");
        int i = 0;
        for (auto z : z_list) {
            // TRACE_SCOPE("th X x");
            for (auto th : theta_list) {
                // TRACE_SCOPE("x");
                for (auto x : x_list) {
                    const T h = results[i++];
                    fprintf(fp, "%f %f %f %f\n", th, x, z, h);
                }
            }
        }
        fclose(fp);
    }

    return 0;
}
