#pragma once

#include <thrust/random.h>

#include <bmp/utils/common.h>
#include <bmp/utils/compute.h>

namespace bmp {

template <typename BitmapType> struct LocateTile64 {
    const BitmapType ncols;
    explicit LocateTile64(BitmapType ncol_tile) : ncols(ncol_tile) {}

    __host__ __device__ thrust::tuple<BitmapType, BitmapType>
    operator()(const thrust::tuple<BitmapType, BitmapType> &indices) const {
        const BitmapType row = thrust::get<0>(indices);
        const BitmapType col = thrust::get<1>(indices);
        BitmapType tile_index = (row / TILE_DIM) * ncols + (col / TILE_DIM);
        BitmapType position =
            (BitmapType)1 << ((row % TILE_DIM) * TILE_DIM + (col % TILE_DIM));

        return thrust::make_tuple(tile_index, position);
    }
};

// Functor for filling row_indices from csr_input.row_pointers
template <typename IndexType> struct FillRowIndices {
    const IndexType *row_pointers;
    IndexType *row_indices;

    explicit FillRowIndices(const IndexType *_row_pointers,
                            IndexType *_row_indices)
        : row_pointers(_row_pointers), row_indices(_row_indices) {}

    __device__ void operator()(const IndexType row) const {
        for (IndexType i = row_pointers[row]; i < row_pointers[row + 1]; ++i) {
            row_indices[i] = row;
        }
    }
};

// Struct for calculating the offset based on tile index.
template <typename IndexType, typename BitmapType> struct COOIndices {
    const BitmapType ncols;
    explicit COOIndices(BitmapType num_cols) : ncols(num_cols) {}

    __host__ __device__ thrust::tuple<IndexType, IndexType>
    operator()(BitmapType tileIndex) const {
        IndexType row = static_cast<IndexType>(tileIndex / ncols);
        IndexType col = static_cast<IndexType>(tileIndex % ncols);
        return thrust::make_tuple(row, col);
    }
};

template <typename IndexType, typename BitmapType> struct CountBits {
    __device__ IndexType operator()(BitmapType bmp64) {
        return (IndexType)__popcll(bmp64);
    }
};

}  // namespace bmp