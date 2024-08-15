#pragma once

#include <cuda_fp16.h>


#define DEBUG 0

namespace bmp {

constexpr uint32_t BALLOT_MASK = 0xffffffff;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_BLOCK = 32;                         // warps per block
constexpr int THREADS_BLOCK = WARPS_BLOCK * WARP_SIZE;  // threads per block
constexpr int TILE64S_WARP = 2;                            // tiles per warp
constexpr int TILE64S_BLOCK = WARPS_BLOCK * TILE64S_WARP;  // tiles per block
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int TILE_DIM = 8;
constexpr int TILE_SIZE = 64;
constexpr int FRAG_DIM = 16;
constexpr int FRAG_SIZE = 256;


using uint8_t = unsigned char;
using bmp64_t = unsigned long long int;
using half = __half;
using half2 = __half2;

__constant__ int frag_offsets[4] = {0, 128, 8, 136};

}  // namespace bmp