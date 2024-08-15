#pragma once
#include <cuda_runtime.h>
#include <chrono>

namespace bmp {

class CUDATimer {
public:
    CUDATimer() {
        cudaEventCreate(&start_time);
        cudaEventCreate(&end_time);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_time);
        cudaEventDestroy(end_time);
    }

    void start() { cudaEventRecord(start_time); }

    void stop() {
        cudaEventRecord(end_time);
        cudaEventSynchronize(end_time);
    }

    float elapsed() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_time, end_time);
        return milliseconds;
    }

private:
    cudaEvent_t start_time;
    cudaEvent_t end_time;
};

class CPUTimer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }

    void stop() { end_time = std::chrono::high_resolution_clock::now(); }

    double elapsed() const {
        return std::chrono::duration<double, std::milli>(end_time - start_time)
            .count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time,
        end_time;
};

}  // namespace bmp