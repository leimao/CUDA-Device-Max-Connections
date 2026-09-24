# CUDA Device Max Connections

This project measures and profiles CUDA work submitted concurrently from CPU
threads. It creates 32 nonblocking CUDA streams, with each host thread
submitting 200 compute-bound kernels to its stream.

The executable runs two phases:

1. A throughput pass that reports total queries, elapsed time, and queries per
   second.
2. A CUDA-only PyTorch Kineto profiling pass that writes a Chrome trace JSON
   file.

Set `CUDA_DEVICE_MAX_CONNECTIONS` before execution to compare how CUDA hardware
work queues affect the submission pattern.

## Requirements

- NVIDIA GPU and driver compatible with the container image
- Docker with NVIDIA Container Toolkit support (`--gpus all`)

The commands below use the NGC PyTorch image, which provides CUDA, CMake,
PyTorch, LibTorch, and Kineto.

## Build

From the repository root on the host, start the container:

```bash
docker run -it --rm --gpus all -v "$(pwd)":/mnt -w /mnt \
  nvcr.io/nvidia/pytorch:26.08-py3
```

Inside the container, configure CMake using PyTorch's CMake package location,
then build with parallel jobs:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j
```

The source is split into independent translation units so CMake can compile
host orchestration and CUDA kernel code in parallel. On incremental builds,
only the translation units affected by an edit need recompilation.

## Run

Run with the CUDA runtime default connection setting:

```bash
./build/multi_stream
```

Run with a specific connection limit:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 ./build/multi_stream
CUDA_DEVICE_MAX_CONNECTIONS=8 ./build/multi_stream
CUDA_DEVICE_MAX_CONNECTIONS=32 ./build/multi_stream
```

Specify the trace output path with `--trace-file`:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 ./build/multi_stream \
  --trace-file build/max_conn_1.json
CUDA_DEVICE_MAX_CONNECTIONS=8 ./build/multi_stream \
  --trace-file build/max_conn_8.json
CUDA_DEVICE_MAX_CONNECTIONS=32 ./build/multi_stream \
  --trace-file build/max_conn_32.json
```

Run all comparison values and write a separate trace for each:

```bash
for connections in 1 2 4 8 16 32; do
  CUDA_DEVICE_MAX_CONNECTIONS="$connections" ./build/multi_stream \
    --trace-file "build/max_conn_${connections}.json"
done
```

Each run prints the measured throughput and writes a trace in the working
directory unless `--trace-file` is provided:

```text
multi_stream_trace_max_conn_default.json
multi_stream_trace_max_conn_1.json
multi_stream_trace_max_conn_8.json
multi_stream_trace_max_conn_32.json
```

## Inspecting Traces

Open a generated JSON trace with `chrome://tracing`, Perfetto, or another
Chrome trace-compatible viewer. Compare the CUDA stream timelines and kernel
overlap across `CUDA_DEVICE_MAX_CONNECTIONS` values.

## Project Layout

- `multi_stream_main.cpp`: program entry point
- `multi_stream_runner.cpp`: stream management, throughput measurement, and
  Kineto profiling
- `cuda_worker.cu`: CUDA delay kernel and per-thread kernel submission
- `cuda_utils.h`: CUDA error-checking helpers
- `CMakeLists.txt`: CMake target definition and LibTorch linkage
