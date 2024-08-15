#pragma once

#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

namespace bmp {

struct host_memory {};
struct device_memory {};

template <typename T, typename MemorySpace> struct VectorTrait;

template <typename T> struct VectorTrait<T, host_memory> {
    using MemoryVector = thrust::host_vector<T>;
    static constexpr auto execution_policy() -> decltype(thrust::host) {
        return thrust::host;
    }
};

template <typename T> struct VectorTrait<T, device_memory> {
    using MemoryVector = thrust::device_vector<T>;
    static constexpr auto execution_policy() -> decltype(thrust::device) {
        return thrust::device;
    }
};

template <typename T, typename MemorySpace>
using VectorType = typename VectorTrait<T, MemorySpace>::MemoryVector;

}  // namespace bmp