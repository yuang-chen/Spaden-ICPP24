#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include <bmp/matrices/vector_traits.h>
#include <bmp/utils/common.h>

namespace bmp {

template <typename IndexType, typename ValueType, typename BitmapType,
          typename MemorySpace>
class BitmapCSR {
public:
    using index_type = IndexType;
    using value_type = ValueType;
    using bitmap_type = BitmapType;

    using IndexVector = VectorType<IndexType, MemorySpace>;
    using ValueVector = VectorType<ValueType, MemorySpace>;
    using BitmapVector = VectorType<BitmapType, MemorySpace>;

    IndexType num_rows{0};
    IndexType num_cols{0};
    IndexType num_tiles{0};
    IndexType num_entries{0};

    IndexVector row_pointers;    // num of tile-rows
    IndexVector column_indices;  // num of tiles
    IndexVector tile_offsets;    // num of tiles+1,  nnz/tile & exclusive_scan
    BitmapVector bitmaps;        // num of tiles
    ValueVector values;          // num_entries = nnz_csr

    BitmapCSR() = default;

    // Constructor with dimensions and default value
    BitmapCSR(IndexType nrow, IndexType ncol, IndexType nnz, IndexType ntile) {
        this->num_rows = nrow;
        this->num_cols = ncol;
        this->num_entries = nnz;
        this->num_tiles = ntile;

        this->row_pointers.resize(nrow + 1);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile + 1);
        this->bitmaps.resize(ntile);
        this->values.resize(nnz);
    }

    // Resize the matrix 256 x 256 (1000) -> 256/8 x 256/8 = 32 x 32 (1000) (16
    // 8x8 tiles)
    void resize(IndexType nrow, IndexType ncol, IndexType nnz,
                IndexType ntile) {
        num_rows = nrow;
        num_cols = ncol;
        num_entries = nnz;
        num_tiles = ntile;

        this->row_pointers.resize(nrow + 1);
        this->column_indices.resize(ntile);
        this->tile_offsets.resize(ntile + 1);
        this->bitmaps.resize(ntile);
        this->values.resize(nnz);
    }
};

}  // namespace bmp