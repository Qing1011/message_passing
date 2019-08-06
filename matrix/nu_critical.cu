#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include <thrust/device_vector.h>

template <typename T> std::vector<T> range(T begin, T end, T step)
{
    std::vector<T> lst;
    for (T x = begin; x < end; x += step) { lst.push_back(x); }
    return lst;
}

template <typename T> __device__ T binom_pmf(int n, T p, int i)
{
    return std::exp(std::lgamma(static_cast<T>(n + 1)) -
                    std::lgamma(static_cast<T>(i + 1)) -
                    std::lgamma(static_cast<T>(n - i + 1))  //
                    + i * std::log(p) + (n - i) * std::log(1 - p));
}

template <typename T> __device__ T poisson_pmf(T p, int k)
{
    return std::exp(k * std::log(p) - p - std::lgamma(static_cast<T>(k + 1)));
}

template <typename T> __device__ float thrshTransExtd(T nu, int k, T th)
{
    T s = 0;
    for (int i = static_cast<int>(th * k); i < k; ++i) {
        s += binom_pmf(k - 1, nu, i);
    }
    return s;
}

template <typename T> __device__ float H(T theta, T nu, T z)
{
    const int k_max = 1000;
    T h = 0;
    for (int k = 1; k <= k_max; ++k) {
        const T p_k = poisson_pmf(z, k);
        h += static_cast<T>(k) / z * p_k * thrshTransExtd(nu, k, theta);
    }
    return h;
}

template <typename T>
__global__ void kern_nu_critical(const T *x_list, const int x_size,          //
                                 const T *z_list, const int z_size,          //
                                 const T *theta_list, const int theta_size,  //
                                 T *results)
{
    const int grid_idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size_x = gridDim.x * blockDim.x;

    const int grid_idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int grid_size_y = gridDim.y * blockDim.y;

    const int grid_idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    const int grid_size_z = gridDim.z * blockDim.z;

    for (int i = grid_idx_x; i < x_size; i += grid_size_x) {
        const T x = x_list[i];
        for (int j = grid_idx_y; j < z_size; j += grid_size_y) {
            const T z = z_list[j];
            for (int k = grid_idx_z; k < theta_size; k += grid_size_z) {
                const T theta = theta_list[k];
                const T h = H(theta, x, z);
                const int idx = (i * z_size + j) * theta_size + k;
                results[idx] = h;
            }
        }
    }
}

int ceil_div(int a, int b) { return (a / b) + (a % b ? 1 : 0); }

template <typename T> const T *data(const thrust::device_vector<T> &t)
{
    return thrust::raw_pointer_cast(&t[0]);
}

template <typename T>
std::vector<T> nu_critical(const std::vector<T> &z_list,
                           const std::vector<T> &theta_list,
                           const std::vector<T> &x_list)
{
    const thrust::device_vector<T> x_list_gpu(x_list);
    const thrust::device_vector<T> z_list_gpu(z_list);
    const thrust::device_vector<T> theta_list_gpu(theta_list);

    const int z_size = z_list.size();
    const int theta_size = theta_list.size();
    const int x_size = x_list.size();

    thrust::device_vector<T> results_gpu(z_size * theta_size * x_size);

    std::cout << "result size: " << results_gpu.size() << std::endl;

    dim3 threadsPerBlock(8, 8, 8);
    dim3 blocksPerGrid(ceil_div(x_size, 8),  //
                       ceil_div(z_size, 8),  //
                       ceil_div(theta_size, 8));

    std::cout << "launching kernel" << std::endl;
    kern_nu_critical<<<blocksPerGrid, threadsPerBlock>>>(
        data(x_list_gpu), x_list_gpu.size(),  //
        data(z_list_gpu), z_list_gpu.size(),  //
        data(theta_list_gpu), theta_list_gpu.size(),
        thrust::raw_pointer_cast(&results_gpu[0]));
    cudaDeviceSynchronize();
    std::cout << "kernel finished" << std::endl;

    const thrust::host_vector<T> results_cpu(results_gpu);

    std::vector<T> results(z_size * theta_size * x_size);
    for (int i = 0; i < results.size(); ++i) { results[i] = results_cpu[i]; }
    return results;
}

template <typename T>
void save_result(const std::vector<T> &z_list,      //
                 const std::vector<T> &theta_list,  //
                 const std::vector<T> &x_list,      //
                 const std::vector<T> &results)
{
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
    using T = float;

    const T z_step = 0.1;
    const T theta_step = 0.01;
    const T x_step = 0.002;

    const auto z_list = range<T>(1, 16, z_step);
    const auto theta_list = range<T>(0.01, 1, theta_step);
    const auto x_list = range<T>(0.001, 1, x_step);

    std::cout << "z list: " << z_list.size() << std::endl;
    std::cout << "theta list: " << theta_list.size() << std::endl;
    std::cout << "x list: " << x_list.size() << std::endl;

    auto results = nu_critical(z_list, theta_list, x_list);
    save_result(z_list, theta_list, x_list, results);
    return 0;
}
