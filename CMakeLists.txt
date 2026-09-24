cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(multi_stream_profiler LANGUAGES CXX CUDA)

# Require C++17 for LibTorch compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find PyTorch / LibTorch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Split host orchestration, the CUDA worker, and the entry point into separate
# translation units so CMake can compile them in parallel and incremental
# edits rebuild only their owning unit.
add_executable(multi_stream
	multi_stream_main.cpp
	multi_stream_runner.cpp
	cuda_worker.cu)

# Link against LibTorch (brings in CUDA targets, cuDNN, CUPTI, etc.)
target_link_libraries(multi_stream "${TORCH_LIBRARIES}")