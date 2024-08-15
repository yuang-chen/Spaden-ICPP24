#pragma once
#include <bmp/utils/common.h>
#include <bmp/utils/timer.h>
#include <cuda.h>
#include <mma.h>

namespace bmp {

template <typename IndexType, typename ValueType, typename BitmapType>
__global__ void
spmv_fff(int C_nrow, int C_nnz, const IndexType *__restrict__ A_rowptr,
         const IndexType *__restrict__ A_colidx,
         const BitmapType *__restrict__ A_bitmap,
         const ValueType *__restrict__ A_values,
         const IndexType *__restrict__ A_offset,
         const ValueType *__restrict__ B_values, ValueType *C_values) {
    const auto wid = threadIdx.x / WARP_SIZE;
    const auto lid = threadIdx.x % WARP_SIZE;
    // Tile ID, not Thread ID
    const auto tid1 = blockIdx.x * TILE64S_BLOCK + wid * TILE64S_WARP;
    const auto tid2 = tid1 + 1;

    const BitmapType lid_times_2 = BitmapType(lid) << 1;
    const BitmapType A_pattern1 = BitmapType(1) << lid_times_2;
    const BitmapType A_pattern2 = BitmapType(2) << lid_times_2;
    const BitmapType A_reverse1 = 64 - lid_times_2;
    const BitmapType A_reverse2 = 63 - lid_times_2;
    const int B_pattern1 = (lid & 3) << 1;
    const int B_pattern2 = B_pattern1 + 1;
    // if (wid == 0) {
    //     printf("lid: %d, B_pattern1: %d, B_pattern2: %d\n", lid, B_pattern1,
    //            B_pattern2);
    // }

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM,
                           half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM,
                           half, nvcuda::wmma::row_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM,
                           FRAG_DIM, float>
        acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, float(0.0));

    auto A_row_begin1 = tid1 < C_nrow ? __ldg(A_rowptr + tid1) : C_nnz;
    auto A_row_begin2 = tid2 < C_nrow ? __ldg(A_rowptr + tid2) : C_nnz;
    auto A_row_end1 = tid1 < C_nrow ? __ldg(A_rowptr + tid1 + 1) : C_nnz;
    auto A_row_end2 = tid2 < C_nrow ? __ldg(A_rowptr + tid2 + 1) : C_nnz;

    auto i1 = A_row_begin1, i2 = A_row_begin2;
    bool process1 = i1 < A_row_end1;
    bool process2 = i2 < A_row_end2;

    if (process1 || process2) {
        nvcuda::wmma::fill_fragment(a_frag, 0);
        nvcuda::wmma::fill_fragment(b_frag, 0);
    }
    while (process1 || process2) {
        if (process1) {
            const BitmapType A_bmp1 = __ldg(A_bitmap + i1);
            const auto B_idx = __ldg(A_colidx + i1);
            const auto offset = __ldg(A_offset + i1);
            const auto pos1 = __popcll(A_bmp1 << A_reverse1);
            const auto pos2 = __popcll(A_bmp1 << A_reverse2);

            a_frag.x[0] = (A_bmp1 & A_pattern1) > 0
                              ? half(__ldg(A_values + offset + pos1))
                              : half(0.0);
            a_frag.x[1] = (A_bmp1 & A_pattern2) > 0
                              ? half(__ldg(A_values + offset + pos2))
                              : half(0.0);
            b_frag.x[0] = B_values[B_idx * TILE_DIM + B_pattern1];
            b_frag.x[1] = B_values[B_idx * TILE_DIM + B_pattern2];

            // thread t0 - 3 get elements e0 - 3 and then sync to the whole
            // warp,
            //     such that thread 0, 4, 8, 12, 16, 20, 24, 28 get elements e0,
            //     ;
            // thread 1, 5, 9, 13, 17, 21, 27, 29 get elements e1 ;
            // thread 2,6,10,14,18,22,28,30 get elements e2 ;
            // thread 3,7,11,15,19,23,29,31 get elements e3;
            // how to use shlf primitives to code the process?
        }
        if (process2) {
            const BitmapType A_bmp2 = __ldg(A_bitmap + i2);
            const auto B_idx = __ldg(A_colidx + i2);
            const auto offset = __ldg(A_offset + i2);
            const auto pos1 = __popcll(A_bmp2 << A_reverse1);
            const auto pos2 = __popcll(A_bmp2 << A_reverse2);

            a_frag.x[6] = (A_bmp2 & A_pattern1) > 0
                              ? half(__ldg(A_values + offset + pos1))
                              : half(0.0);
            a_frag.x[7] = (A_bmp2 & A_pattern2) > 0
                              ? half(__ldg(A_values + offset + pos2))
                              : half(0.0);
            b_frag.x[6] = B_values[B_idx * TILE_DIM + B_pattern1];
            b_frag.x[7] = B_values[B_idx * TILE_DIM + B_pattern2];
        }

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        ++i1;
        ++i2;
        process1 = i1 < A_row_end1;
        process2 = i2 < A_row_end2;
    }

    if (lid % 4 == 0) {
        if (tid1 < C_nrow) {
            const auto C_offset = tid1 * TILE_DIM + lid / 4;
            C_values[C_offset] = acc_frag.x[0];
        }
        if (tid2 < C_nrow) {
            const auto C_offset = tid2 * TILE_DIM + lid / 4;
            C_values[C_offset] = acc_frag.x[6];
        }
    }
}

template <typename IndexType, typename ValueType, typename BitmapType>
__global__ void
spmv_hhf(int C_nrow, int C_nnz, const IndexType *__restrict__ A_rowptr,
         const IndexType *__restrict__ A_colidx,
         const BitmapType *__restrict__ A_bitmap,
         const ValueType *__restrict__ A_values,
         const IndexType *__restrict__ A_offset,
         const ValueType *__restrict__ B_values, float *C_values) {
    const auto wid = threadIdx.x / WARP_SIZE;
    const auto lid = threadIdx.x % WARP_SIZE;
    // Tile ID, not Thread ID
    const auto tid1 = blockIdx.x * TILE64S_BLOCK + wid * TILE64S_WARP;
    const auto tid2 = tid1 + 1;

    const BitmapType lid_times_2 = BitmapType(lid) << 1;
    const BitmapType A_pattern1 = BitmapType(1) << lid_times_2;
    const BitmapType A_pattern2 = BitmapType(2) << lid_times_2;
    const BitmapType A_reverse1 = 64 - lid_times_2;
    const BitmapType A_reverse2 = 63 - lid_times_2;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM,
                           half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM,
                           half, nvcuda::wmma::row_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM,
                           FRAG_DIM, float>
        acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, half(0.0));

    auto A_row_begin1 = tid1 < C_nrow ? __ldg(A_rowptr + tid1) : C_nnz;
    auto A_row_begin2 = tid2 < C_nrow ? __ldg(A_rowptr + tid2) : C_nnz;
    auto A_row_end1 = tid1 < C_nrow ? __ldg(A_rowptr + tid1 + 1) : C_nnz;
    auto A_row_end2 = tid2 < C_nrow ? __ldg(A_rowptr + tid2 + 1) : C_nnz;

    auto i1 = A_row_begin1, i2 = A_row_begin2;
    bool process1 = i1 < A_row_end1;
    bool process2 = i2 < A_row_end2;

    if (process1 || process2) {
        nvcuda::wmma::fill_fragment(a_frag, 0);
        nvcuda::wmma::fill_fragment(b_frag, 0);
    }
    while (process1 || process2) {
        if (process1) {
            const BitmapType A_bmp1 = __ldg(A_bitmap + i1);
            const auto B_idx = __ldg(A_colidx + i1);
            const auto offset = __ldg(A_offset + i1);
            const auto pos1 = __popcll(A_bmp1 << A_reverse1);
            const auto pos2 = __popcll(A_bmp1 << A_reverse2);
            const auto A_idx1 = offset + pos1;
            const auto A_idx2 = offset + pos2;

            // half2 A_val1 = __ldg(A_values + A_idx1 / 2);
            // half2 A_val2 = __ldg(A_values + A_idx2 / 2);
            // half A_val_h1 = A_idx1 % 2 == 0 ? A_val1.x : A_val1.y;
            // half A_val_h2 = A_idx2 % 2 == 0 ? A_val2.x : A_val2.y;

            a_frag.x[0] = (A_bmp1 & A_pattern1) > 0
                              ? load_half(A_values, A_idx1)
                              : half(0.0);
            a_frag.x[1] = (A_bmp1 & A_pattern2) > 0
                              ? load_half(A_values, A_idx2)
                              : half(0.0);

            half2 B_val = __ldg(B_values + B_idx * 4 + (lid & 3));
            b_frag.x[0] = B_val.x;
            b_frag.x[1] = B_val.y;

            // thread t0 - 7 get elements e0 - 7 and then sync to the whole
            // warp,
            //     such that thread 0, 4, 8, 12, 16, 20, 24, 28 get elements e0,
            //     1;
            // thread 1, 5, 9, 13, 17, 21, 27, 29 get elements e2, 3;
            // thread 2,6,10,14,18,22,28,30 get elements e4, 5;
            // thread 3,7,11,15,19,23,29,31 get elements e6, 7;
            // how to use shlf primitives to code the process?
        }
        if (process2) {
            const BitmapType A_bmp2 = __ldg(A_bitmap + i2);
            const auto B_idx = __ldg(A_colidx + i2);
            const auto offset = __ldg(A_offset + i2);
            const auto pos1 = __popcll(A_bmp2 << A_reverse1);
            const auto pos2 = __popcll(A_bmp2 << A_reverse2);
            const auto A_idx1 = offset + pos1;
            const auto A_idx2 = offset + pos2;

            // half2 A_val1 = __ldg(A_values + A_idx1 / 2);
            // half2 A_val2 = __ldg(A_values + A_idx2 / 2);
            // half A_val_h1 = A_idx1 % 2 == 0 ? A_val1.x : A_val1.y;
            // half A_val_h2 = A_idx2 % 2 == 0 ? A_val2.x : A_val2.y;

            a_frag.x[6] = (A_bmp2 & A_pattern1) > 0
                              ? load_half(A_values, A_idx1)
                              : half(0.0);
            a_frag.x[7] = (A_bmp2 & A_pattern2) > 0
                              ? load_half(A_values, A_idx2)
                              : half(0.0);

            half2 B_val = __ldg(B_values + B_idx * 4 + (lid & 3));
            b_frag.x[6] = B_val.x;
            b_frag.x[7] = B_val.y;
        }

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        ++i1;
        ++i2;
        process1 = i1 < A_row_end1;
        process2 = i2 < A_row_end2;
    }

    if (lid % 4 == 0) {
        if (tid1 < C_nrow) {
            const auto C_offset = tid1 * TILE_DIM + lid / 4;
            C_values[C_offset] = acc_frag.x[0];
        }
        if (tid2 < C_nrow) {
            const auto C_offset = tid2 * TILE_DIM + lid / 4;
            C_values[C_offset] = acc_frag.x[6];
        }
    }
}

template <typename IndexType, typename ValueType, typename BitmapType>
__global__ void
spmv_bhf(int C_nrow, int C_nnz, const IndexType *__restrict__ A_rowptr,
         const IndexType *__restrict__ A_colidx,
         const BitmapType *__restrict__ A_bitmap,
         const ValueType *__restrict__ A_values,
         const IndexType *__restrict__ A_offset,
         const ValueType *__restrict__ B_values, float *C_values) {
    const auto wid = threadIdx.x / WARP_SIZE;
    const auto lid = threadIdx.x % WARP_SIZE;
    // Tile ID, not Thread ID
    const auto tid1 = blockIdx.x * TILE64S_BLOCK + wid * TILE64S_WARP;
    const auto tid2 = tid1 + 1;

    const BitmapType lid_times_2 = BitmapType(lid) << 1;
    const BitmapType A_pattern1 = BitmapType(1) << lid_times_2;
    const BitmapType A_pattern2 = BitmapType(2) << lid_times_2;
    const BitmapType A_reverse1 = 64 - lid_times_2;
    const BitmapType A_reverse2 = 63 - lid_times_2;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, FRAG_DIM, FRAG_DIM, FRAG_DIM,
                           half, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, FRAG_DIM, FRAG_DIM, FRAG_DIM,
                           half, nvcuda::wmma::row_major>
        b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, FRAG_DIM, FRAG_DIM,
                           FRAG_DIM, float>
        acc_frag;

    nvcuda::wmma::fill_fragment(acc_frag, half(0.0));

    auto A_row_begin1 = tid1 < C_nrow ? __ldg(A_rowptr + tid1) : C_nnz;
    auto A_row_begin2 = tid2 < C_nrow ? __ldg(A_rowptr + tid2) : C_nnz;
    auto A_row_end1 = tid1 < C_nrow ? __ldg(A_rowptr + tid1 + 1) : C_nnz;
    auto A_row_end2 = tid2 < C_nrow ? __ldg(A_rowptr + tid2 + 1) : C_nnz;

    auto i1 = A_row_begin1, i2 = A_row_begin2;
    bool process1 = i1 < A_row_end1;
    bool process2 = i2 < A_row_end2;

    if (process1 || process2) {
        nvcuda::wmma::fill_fragment(a_frag, 0);
        nvcuda::wmma::fill_fragment(b_frag, 0);
    }
    while (process1 || process2) {
        if (process1) {
            const BitmapType A_bmp1 = __ldg(A_bitmap + i1);
            const auto B_idx = __ldg(A_colidx + i1);
            const auto offset = __ldg(A_offset + i1);
            const auto pos1 = __popcll(A_bmp1 << A_reverse1);
            const auto pos2 = __popcll(A_bmp1 << A_reverse2);
            const auto A_idx1 = offset + pos1;
            const auto A_idx2 = offset + pos2;

            // half2 A_val1 = __ldg(A_values + A_idx1 / 2);
            // half2 A_val2 = __ldg(A_values + A_idx2 / 2);
            // half A_val_h1 = A_idx1 % 2 == 0 ? A_val1.x : A_val1.y;
            // half A_val_h2 = A_idx2 % 2 == 0 ? A_val2.x : A_val2.y;

            a_frag.x[0] = (A_bmp1 & A_pattern1) > 0;  // 1 bit -> 16-bit half
            a_frag.x[1] = (A_bmp1 & A_pattern2) > 0;

            half2 B_val = __ldg(B_values + B_idx * 4 + (lid & 3));
            b_frag.x[0] = B_val.x;
            b_frag.x[1] = B_val.y;

            // thread t0 - 7 get elements e0 - 7 and then sync to the whole
            // warp,
            //     such that thread 0, 4, 8, 12, 16, 20, 24, 28 get elements e0,
            //     1;
            // thread 1, 5, 9, 13, 17, 21, 27, 29 get elements e2, 3;
            // thread 2,6,10,14,18,22,28,30 get elements e4, 5;
            // thread 3,7,11,15,19,23,29,31 get elements e6, 7;
            // how to use shlf primitives to code the process?
        }
        if (process2) {
            const BitmapType A_bmp2 = __ldg(A_bitmap + i2);
            const auto B_idx = __ldg(A_colidx + i2);
            const auto offset = __ldg(A_offset + i2);
            const auto pos1 = __popcll(A_bmp2 << A_reverse1);
            const auto pos2 = __popcll(A_bmp2 << A_reverse2);
            const auto A_idx1 = offset + pos1;
            const auto A_idx2 = offset + pos2;

            // half2 A_val1 = __ldg(A_values + A_idx1 / 2);
            // half2 A_val2 = __ldg(A_values + A_idx2 / 2);
            // half A_val_h1 = A_idx1 % 2 == 0 ? A_val1.x : A_val1.y;
            // half A_val_h2 = A_idx2 % 2 == 0 ? A_val2.x : A_val2.y;

            a_frag.x[6] = (A_bmp2 & A_pattern1) > 0;
            a_frag.x[7] = (A_bmp2 & A_pattern2) > 0;

            half2 B_val = __ldg(B_values + B_idx * 4 + (lid & 3));
            b_frag.x[6] = B_val.x;
            b_frag.x[7] = B_val.y;
        }

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        ++i1;
        ++i2;
        process1 = i1 < A_row_end1;
        process2 = i2 < A_row_end2;
    }

    if (lid % 4 == 0) {
        if (tid1 < C_nrow) {
            const auto C_offset = tid1 * TILE_DIM + lid / 4;
            C_values[C_offset] = acc_frag.x[0];
        }
        if (tid2 < C_nrow) {
            const auto C_offset = tid2 * TILE_DIM + lid / 4;
            C_values[C_offset] = acc_frag.x[6];
        }
    }
}

__global__ void spmv_csr_warp16_kernel(const int *rowPtr, const int *colIndices,
                                       const float *values, const float *x,
                                       float *y, int numRows) {
    int warpId = threadIdx.x / 32;  // Assuming blockDim.x is a multiple of 32
    int laneId = threadIdx.x % 32;  // Lane within the warp
    int row = (blockIdx.x * (blockDim.x / 32) + warpId) *
              16;  // Starting row for this warp

    // Each thread within the warp processes two rows
    for (int r = 0; r < 16; r += 2) {
        int actualRow = row + r;
        if (actualRow < numRows) {  // Check if the row is within bounds
            float sum = 0;
            // Process the first row of the pair
            int rowStart = rowPtr[actualRow];
            int rowEnd = rowPtr[actualRow + 1];
            for (int j = rowStart + laneId; j < rowEnd; j += 32) {
                sum += values[j] * x[colIndices[j]];
            }
            // Reduce within the warp to sum up the partial results
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

            // Only the first thread of each half-warp writes the result
            if (laneId == 0)
                y[actualRow] = sum;

            // Process the second row of the pair (if within bounds)
            actualRow += 1;
            if (actualRow < numRows) {
                sum = 0;
                rowStart = rowPtr[actualRow];
                rowEnd = rowPtr[actualRow + 1];
                for (int j = rowStart + laneId; j < rowEnd; j += 32) {
                    sum += values[j] * x[colIndices[j]];
                }
                // Reduce within the warp
                for (int offset = 16; offset > 0; offset /= 2)
                    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

                if (laneId == 0)
                    y[actualRow] = sum;
            }
        }
    }
}
template <typename Config, typename CsrMatrix>
void spmv_csr_warp16(const Config &config, const CsrMatrix &A,
                     const thrust::device_vector<float> &x,
                     thrust::device_vector<float> &y) {
    int blockSize = 512;
    int gridSize = (A.num_entries + 15) / 16;
    auto A_rowptr_ptr = thrust::raw_pointer_cast(A.row_pointers.data());
    auto A_values_ptr = thrust::raw_pointer_cast(A.values.data());
    auto A_colidx_ptr = thrust::raw_pointer_cast(A.column_indices.data());
    auto B_values_ptr = thrust::raw_pointer_cast(x.data());
    auto C_values_ptr = thrust::raw_pointer_cast(y.data());

    CUDATimer timer;
    timer.start();
    for (int i = 0; i < config.exec_iterations; i++)
        spmv_csr_warp16_kernel<<<gridSize, blockSize>>>(
            A_rowptr_ptr, A_colidx_ptr, A_values_ptr, B_values_ptr,
            C_values_ptr, A.num_rows);

    timer.stop();
    auto spmv_time = timer.elapsed() / config.exec_iterations;
    auto gflops = A.values.size() / spmv_time / 1e6;
    printf("spmv_warp16: %8.4lf ms, %8.4lf Gflops\n", spmv_time, gflops);
}

}  // namespace bmp