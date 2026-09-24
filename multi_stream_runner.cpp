#include "multi_stream.h"

#include "cuda_utils.h"
#include "cuda_worker.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

int run_multi_stream(int argc, char** argv)
{
    char const* max_conn_env = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    std::string max_conn_str = max_conn_env ? max_conn_env : "default";
    std::string trace_filename =
        "multi_stream_trace_max_conn_" + max_conn_str + ".json";

    if (argc == 3 && std::string(argv[1]) == "--trace-file")
    {
        trace_filename = argv[2];
    }
    else if (argc != 1)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [--trace-file <path-to-trace.json>]" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "CUDA_DEVICE_MAX_CONNECTIONS = "
              << (max_conn_env ? max_conn_env : "Not Set (Defaults to 8)")
              << std::endl;

    int const num_threads = 32;
    int const queries_per_thread = 200;
    unsigned long long const cycles = 1000000ULL;
    int const total_queries = num_threads * queries_per_thread;

    std::vector<cudaStream_t> streams(num_threads);
    for (int thread_index = 0; thread_index < num_threads; ++thread_index)
    {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&streams[thread_index],
                                                   cudaStreamNonBlocking));
    }

    at::ThreadLocalState tls_state;

    std::cout << "\n--- Phase 1: Measuring Pure System Throughput (QPS) ---"
              << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> benchmark_workers;
    for (int thread_index = 0; thread_index < num_threads; ++thread_index)
    {
        benchmark_workers.emplace_back(enqueue_delay_kernels,
                                       streams[thread_index],
                                       queries_per_thread, cycles, tls_state);
    }

    for (auto& worker : benchmark_workers)
    {
        worker.join();
    }

    for (cudaStream_t stream : streams)
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double elapsed_seconds = elapsed.count();
    double throughput = total_queries / elapsed_seconds;

    std::cout << "Total Queries Processed : " << total_queries << std::endl;
    std::cout << "Elapsed Time            : " << elapsed_seconds << " seconds"
              << std::endl;
    std::cout << "Pure System Throughput  : " << throughput << " queries/second"
              << std::endl;

    std::cout << "\n--- Phase 2: Collecting Profiling Trace ---" << std::endl;
    torch::autograd::profiler::ProfilerConfig profiler_config(
        torch::autograd::profiler::ProfilerState::KINETO,
        /*report_input_shapes=*/false,
        /*profile_memory=*/false,
        /*with_stack=*/false,
        /*with_flops=*/false,
        /*with_modules=*/false);
    std::set<torch::autograd::profiler::ActivityType> activities = {
        torch::autograd::profiler::ActivityType::CUDA};

    torch::autograd::profiler::prepareProfiler(profiler_config, activities);
    torch::autograd::profiler::enableProfiler(profiler_config, activities);

    std::vector<std::thread> profile_workers;
    for (int thread_index = 0; thread_index < num_threads; ++thread_index)
    {
        profile_workers.emplace_back(enqueue_delay_kernels,
                                     streams[thread_index], queries_per_thread,
                                     cycles, tls_state);
    }

    for (auto& worker : profile_workers)
    {
        worker.join();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto profiler_result = torch::autograd::profiler::disableProfiler();

    for (cudaStream_t stream : streams)
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    if (profiler_result)
    {
        profiler_result->save(trace_filename);
        std::cout << "Saved PyTorch Profiler trace to: " << trace_filename
                  << std::endl;
    }

    for (cudaStream_t stream : streams)
    {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }

    return 0;
}