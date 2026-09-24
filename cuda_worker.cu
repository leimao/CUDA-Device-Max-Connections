#include "cuda_worker.h"

#include "cuda_utils.h"

__global__ void delay_kernel(unsigned long long delay_cycles)
{
    unsigned long long start = clock64();
    while (clock64() - start < delay_cycles)
    {
    }
}

void enqueue_delay_kernels(cudaStream_t stream, int num_queries,
                           unsigned long long delay_cycles,
                           at::ThreadLocalState tls_state)
{
    at::ThreadLocalStateGuard guard(tls_state);

    for (int query = 0; query < num_queries; ++query)
    {
        delay_kernel<<<1, 128, 0, stream>>>(delay_cycles);
        CHECK_LAST_CUDA_ERROR();
    }
}