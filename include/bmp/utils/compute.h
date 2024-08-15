
#pragma once
#include <cuda_fp16.h>

#include <bmp/utils/common.h>

namespace bmp {
__device__ __forceinline__ half load_half(const half2 *ptr, int idx) {
    half2 A_val1 = __ldg(ptr + idx / 2);
    return idx % 2 == 0 ? A_val1.x : A_val1.y;
}

/**
 * @brief Helper to round up to the nearest multiple of 'r'.
 */
constexpr __host__ __device__ __forceinline__ int round_up(int x, int r) {
    return (x + r - 1) / r * r;
}

/**
 * @brief Dividy x by y and round up.
 */
constexpr __host__ __device__ __forceinline__ int div_up(int x, int y) {
    return (x + y - 1) / y;
}

/**
 * @brief Compute log base 2 statically. Only works when x
 * is a power of 2 and positive.
 *
 * TODO(tgale): GCC doesn't like this function being constexpr. Ensure
 * that this is evaluated statically.
 */
__host__ __device__ __forceinline__ int log2(int x) {
    if (x >>= 1)
        return log2(x) + 1;
    return 0;
}

/**
 * @brief Find the minimum statically.
 */
constexpr __host__ __device__ __forceinline__ int min(int a, int b) {
    return a < b ? a : b;
}

/**
 * @brief Find the maximum statically.
 */
constexpr __host__ __device__ __forceinline__ int max(int a, int b) {
    return a > b ? a : b;
}

template <typename BitmapType> inline constexpr size_t Dimension() {
    static_assert(std::is_same<BitmapType, bmp64_t>::value,
                  "BitmapType must be an bmp64_t.");
    return 8;
};

template <typename T> __device__ __forceinline__ void swap(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}

}  // namespace bmp