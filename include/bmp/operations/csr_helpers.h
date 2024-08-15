#pragma once

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

#include <bmp/utils/functors.h>

namespace bmp {

template <typename IndexType>
void get_row_lengths_from_pointers(
    thrust::device_vector<IndexType> &row_lens,
    const thrust::device_vector<IndexType> &row_ptrs) {
    thrust::transform(row_ptrs.begin() + 1, row_ptrs.end(), row_ptrs.begin(),
                      row_lens.begin(), thrust::minus<IndexType>());
}

template <typename IndexType>
void get_row_pointers_from_indices(
    thrust::device_vector<IndexType> &row_pointers,
    const thrust::device_vector<IndexType> &row_indices) {
    assert(thrust::is_sorted(row_indices.begin(), row_indices.end()) &&
           "row_indices must be sorted");

    thrust::lower_bound(
        row_indices.begin(), row_indices.end(),
        thrust::counting_iterator<IndexType>(0),
        thrust::counting_iterator<IndexType>(row_pointers.size()),
        row_pointers.begin());
}

template <typename IndexVector, typename ValueVector>
void sort_columns_per_row(IndexVector &row_indices, IndexVector &column_indices,
                          ValueVector &values) {
    // sort columns per row
    thrust::sort_by_key(column_indices.begin(), column_indices.end(),
                        thrust::make_zip_iterator(thrust::make_tuple(
                            row_indices.begin(), values.begin())));
    thrust::stable_sort_by_key(row_indices.begin(), row_indices.end(),
                               thrust::make_zip_iterator(thrust::make_tuple(
                                   column_indices.begin(), values.begin())));
}

template <typename IndexType>
void get_row_indices_from_pointers(
    const thrust::device_vector<IndexType> &row_pointers,
    thrust::device_vector<IndexType> &row_indices) {
    const auto num_rows = row_pointers.size() - 1;

    thrust::for_each(thrust::counting_iterator<IndexType>(0),
                     thrust::counting_iterator<IndexType>(num_rows),
                     FillRowIndices<IndexType>(
                         thrust::raw_pointer_cast(row_pointers.data()),
                         thrust::raw_pointer_cast(row_indices.data())));
}

}  // namespace bmp