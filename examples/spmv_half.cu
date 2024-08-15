#include <bmp/bmp.h>

using namespace bmp;

template <typename BmpMat>
void execute_spmv(const Config &config, const BmpMat &A,
                  const thrust::device_vector<half2> &B_values, /*B*/
                  thrust::device_vector<float> &C_values /*C*/) {
    auto A_rowptr_ptr = thrust::raw_pointer_cast(A.row_pointers.data());
    auto A_colidx_ptr = thrust::raw_pointer_cast(A.column_indices.data());
    auto A_offset_ptr = thrust::raw_pointer_cast(A.tile_offsets.data());
    auto A_bitmap_ptr = thrust::raw_pointer_cast(A.bitmaps.data());
    auto A_values_ptr = thrust::raw_pointer_cast(A.values.data());
    auto B_values_ptr = thrust::raw_pointer_cast(B_values.data());
    auto C_values_ptr = thrust::raw_pointer_cast(C_values.data());

    std::string tag = "[bmp]";

    int gridDim = div_up(A.num_rows, TILE64S_BLOCK);
    int blockDim = TILE64S_WARP * WARP_SIZE;
    CUDATimer timer;

    timer.start();

    for (int i = 0; i < config.exec_iterations; i++) {
        assert((A.num_cols * TILE_DIM) == (B_values.size() * 2));

        spmv_hhf<<<gridDim, blockDim>>>(
            A.num_rows, A.num_entries, A_rowptr_ptr, A_colidx_ptr, A_bitmap_ptr,
            A_values_ptr, A_offset_ptr, B_values_ptr, C_values_ptr);
    }
    timer.stop();
    auto spmv_time = timer.elapsed() / config.exec_iterations;
    auto gflops = A.values.size() / spmv_time / 1e6;
    printf("%s spmv_hhf: %8.4lf ms, %8.4lf Gflops\n", tag.c_str(), spmv_time,
           gflops);

    timer.start();
    for (int i = 0; i < config.exec_iterations; i++) {
        spmv_bhf<<<gridDim, blockDim>>>(
            A.num_rows, A.num_entries, A_rowptr_ptr, A_colidx_ptr, A_bitmap_ptr,
            A_values_ptr, A_offset_ptr, B_values_ptr, C_values_ptr);
    }
    timer.stop();
    spmv_time = timer.elapsed() / config.exec_iterations;
    gflops = A.values.size() / spmv_time / 1e6;
    printf("%s spmv_bhf: %8.4lf ms, %8.4lf Gflops\n", tag.c_str(), spmv_time,
           gflops);
}

int main(int argc, char **argv) {
    cudaSetDevice(0);

    // XXX !!!!!! CSR format doesn't work with symmetric matrix marker format
    // !!!!!!
    // DCoo A_csr;

    CsrMatrix<int, float, device_memory> A_csr;

    Config config = program_options(argc, argv);

    if (!config.input_file.empty()) {
        read_matrix_file(A_csr, config.input_file);
    }

    auto nrow = A_csr.num_rows;
    auto nnz = A_csr.num_entries;
    printf(
        "matrix A has shape (%d x %d) and %d entries, with density %lf and nnz/nrow = %d\n",
        nrow, nrow, nnz, static_cast<double>(nnz) / nrow / nrow, nnz / nrow);

    thrust::device_vector<float> d_vec_B(nrow, 1.0);
    thrust::device_vector<float> d_vec_C(nrow, 0);
    //<<<<<<<<<<<<<<<<<<
    //<< benchmark SpMV
    //<<<<<<<<<<<<<<<<<

    //>>>>>>>>>>
    //>> bmp SpMV
    //>>>>>>>>>>
    printf("\n--------------bitmap----------------\n");

    //? A_bmp
    BitmapCSR<int, half2, bmp64_t, device_memory> A_bmp;
    CUDATimer timer;
    timer.start();
    convert_csr2bmp(A_csr, A_bmp);
    timer.stop();
    printf("conversion time: %f ms\n", timer.elapsed());

    // print_vec(A_bmp.row_pointers, "A row_pointers");
    // print_vec(A_bmp.column_indices, "A column_indices");

    //? vec_B
    const auto nrow_aligned = A_bmp.num_rows * TILE_DIM;
    thrust::device_vector<half2> B_dense(nrow_aligned / 2);
    copy_float_to_half2(d_vec_B, B_dense);

    //? d_bmp_vec_C
    thrust::device_vector<float> C_dense(nrow_aligned);

    execute_spmv(config, A_bmp, B_dense, C_dense);

    return 0;
}
