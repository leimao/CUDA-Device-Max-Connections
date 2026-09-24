#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

inline void check_cuda_error(cudaError_t error, char const* expression,
                             char const* file, int line)
{
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(error) << " " << expression
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void check_last_cuda_error(char const* file, int line)
{
    cudaError_t const error{cudaGetLastError()};
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(value)                                                \
    check_cuda_error((value), #value, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() check_last_cuda_error(__FILE__, __LINE__)

#endif // CUDA_UTILS_H