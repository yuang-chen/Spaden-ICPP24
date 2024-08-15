#pragma once

#include <thrust/unique.h>

#include <bmp/operations/csr_helpers.h>
#include <bmp/utils/compute.h>
#include <bmp/utils/functors.h>
#include "csr_helpers.h"

namespace bmp {

__global__ void float_to_half2_kernel(const float *input, half2 *output,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure we don't access out of bounds and handle the case where
    // size is odd.
    if (idx * 2 < size) {
        float f1 = input[idx * 2];
        float f2 = (idx * 2 + 1) < size ? input[idx * 2 + 1] : 0.0f;
        // Low 16 bits of the return value correspond to the input a, high 16
        // bits correspond to the input b.

        output[idx] = __floats2half2_rn(f1, f2);
    }
}

void copy_float_to_half2(const auto &input,
                         thrust::device_vector<half2> &output) {
    int GridDim = div_up(output.size(), 256);
    int BlockDim = 256;
    float_to_half2_kernel<<<GridDim, BlockDim>>>(
        input.data().get(), output.data().get(), input.size());
}

template <typename CsrMatrix, typename BitCSR64>
void convert_csr2bmp(CsrMatrix mat_input, BitCSR64 &mat_output) {
    static_assert(std::is_same_v<typename CsrMatrix::value_type, float>);
    using IndexType = typename BitCSR64::index_type;
    using ValueType = typename BitCSR64::value_type;
    using BitmapType = typename BitCSR64::bitmap_type;
    // Use thrust::device directly for simplicity and readability.

    auto exec = thrust::device;
    const auto nnz = mat_input.num_entries;
    const auto nrow = mat_input.num_rows;
    const auto ncol = mat_input.num_cols;
    const auto nrow_tile = div_up(nrow, TILE_DIM);
    const auto ncol_tile = div_up(ncol, TILE_DIM);
    assert(nrow_tile * ncol_tile < std::numeric_limits<BitmapType>::max() &&
           "BitmapType is not large enough to represent the number of tiles");
    // Create a device vector for row_indices, since CSR doesn't explicitly
    // store them.
    // printf("nrow_tile: %d\n", nrow_tile);
    thrust::device_vector<IndexType> row_indices(nnz);

    // Fill row_indices based on csr_input.row_pointers.
    get_row_indices_from_pointers(mat_input.row_pointers, row_indices);

    sort_columns_per_row(row_indices, mat_input.column_indices,
                         mat_input.values);
    // print_vec(row_indices.end() - 10, row_indices.end(), "row_indices: ",
    // 10); print_vec(mat_input.column_indices.end() - 10,
    //           mat_input.column_indices.end(), "column_indices: ", 10);
    thrust::device_vector<BitmapType> tile_indices(nnz);
    thrust::device_vector<BitmapType> pos_in_tile(nnz);
    // print_vec(tile_indices, "0 tile_indices: ", 10);
    // print_vec(tile_indices.end() - 10, tile_indices.end(),
    //           "0 tile_indices: ", 10);
    // Calculate tile indices and pos_in_tiles with a single pass.
    thrust::transform(
        exec,
        thrust::make_zip_iterator(thrust::make_tuple(
            row_indices.begin(), mat_input.column_indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            row_indices.end(), mat_input.column_indices.end())),
        thrust::make_zip_iterator(
            thrust::make_tuple(tile_indices.begin(), pos_in_tile.begin())),
        LocateTile64<BitmapType>(ncol_tile));

    // print_vec(tile_indices, "1 tile_indices: ", 10);
    // print_vec(tile_indices.end() - 10, tile_indices.end(),
    //           "1 tile_indices: ", 10);

    // Sort based on tile indices. This operation affects the original matrices
    // in-place.
    //! due to this step, we have to utilize a vector of row_indices
    thrust::stable_sort_by_key(
        exec, tile_indices.begin(), tile_indices.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            row_indices.begin(), mat_input.column_indices.begin(),
            mat_input.values.begin(), pos_in_tile.begin())));
    // print_vec(tile_indices, "2 tile_indices: ", 10);

    // Perform reduction by key in-place where possible.
    // Using Thrust's reduce_by_key to compact and aggregate  pos_in_tiles.
    thrust::device_vector<BitmapType> unique_tile_indices(nnz);
    thrust::device_vector<BitmapType> bitmap(nnz);

    auto [unique_tile_indices_end, bitmap_end] = thrust::reduce_by_key(
        exec, tile_indices.begin(), tile_indices.end(), pos_in_tile.begin(),
        unique_tile_indices.begin(), bitmap.begin(),
        thrust::equal_to<BitmapType>(), thrust::bit_or<BitmapType>());

    IndexType num_tiles =
        thrust::distance(unique_tile_indices.begin(), unique_tile_indices_end);
    // printf("1 num_tiles: %d\n", num_tiles);
    unique_tile_indices.resize(num_tiles);
    row_indices.resize(num_tiles);
    thrust::fill(row_indices.begin(), row_indices.end(), 0);
    // tile_indices.resize(num_tiles);
    bitmap.resize(num_tiles);
    // Setup output matrix dimensions based on TILE_DIM.

    mat_output.resize(nrow_tile, ncol_tile, nnz, num_tiles);
    // Transform tile indices to row and column indices for the output matrix.
    // print_vec(unique_tile_indices, "unique_tile_indices: ", 10);
    thrust::transform(
        exec, unique_tile_indices.begin(), unique_tile_indices.end(),
        thrust::make_zip_iterator(thrust::make_tuple(
            row_indices.begin(), mat_output.column_indices.begin())),
        COOIndices<IndexType, BitmapType>(ncol_tile));

    // convert row indices to row pointers for CSR
    get_row_pointers_from_indices(mat_output.row_pointers, row_indices);
    thrust::device_vector<int> row_len(nrow_tile);
    get_row_lengths_from_pointers(row_len, mat_output.row_pointers);

    // Copying values and computing bmp_offsets is already efficient.
    if constexpr (std::is_same_v<ValueType, float>) {
        mat_output.values = std::move(mat_input.values);
    } else if constexpr (std::is_same_v<ValueType, half2>) {
        const auto num_elem = div_up(nnz, 2);
        mat_output.values.resize(num_elem);
        copy_float_to_half2(mat_input.values, mat_output.values);
    } else {
        printf("Unsupported ValueType\n");
        std::exit(1);
    }

    // mat_output.tile_offsets.resize(num_tiles);
    thrust::transform(exec, bitmap.begin(), bitmap.end(),
                      mat_output.tile_offsets.begin(),
                      CountBits<IndexType, BitmapType>());

    // Convert population counts to offsets for the bitmap.
    thrust::exclusive_scan(exec, mat_output.tile_offsets.begin(),
                           mat_output.tile_offsets.end(),
                           mat_output.tile_offsets.begin(), 0);

    mat_output.bitmaps = std::move(bitmap);
}

}  // namespace bmp