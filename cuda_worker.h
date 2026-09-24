#ifndef CUDA_WORKER_H
#define CUDA_WORKER_H

#include <ATen/ThreadLocalState.h>
#include <cuda_runtime.h>

void enqueue_delay_kernels(cudaStream_t stream, int num_queries,
                           unsigned long long delay_cycles,
                           at::ThreadLocalState tls_state);

#endif // CUDA_WORKER_H