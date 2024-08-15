#include <bmp/bmp.h>

using namespace bmp;

template <typename BmpMat>
void execute_spmv(const Config &config, const BmpMat &A,
                  const thrust::device_vector<float> &B_values, /*B*/
                  thrust::device_vector<float> &C_values /*C*/) {
    auto A_rowptr_ptr = thrust::raw_pointer_cast(A.row_pointers.data());
    // auto A_rowidx_ptr = thrust::raw_pointer_cast(A.row_indices.data());
    auto A_colidx_ptr = thrust::raw_pointer_cast(A.column_indices.data());
    auto A_offset_ptr = thrust::raw_pointer_cast(A.tile_offsets.data());
    auto A_bitmap_ptr = thrust::raw_pointer_cast(A.bitmaps.data());
    auto A_values_ptr = thrust::raw_pointer_cast(A.values.data());
    auto B_values_ptr = thrust::raw_pointer_cast(B_values.data());
    auto C_values_ptr = thrust::raw_pointer_cast(C_values.data());

    int gridDim = div_up(A.num_rows, TILE64S_BLOCK);
    int blockDim = TILE64S_WARP * WARP_SIZE;
    CUDATimer timer;

    timer.start();

    for (int i = 0; i < config.exec_iterations; i++) {
        assert(A.num_cols * TILE_DIM == B_values.size());

        spmv_fff<<<gridDim, blockDim>>>(

            A.num_rows, A.num_entries, A_rowptr_ptr, A_colidx_ptr, A_bitmap_ptr,
            A_values_ptr, A_offset_ptr, B_values_ptr, C_values_ptr);
    }
    timer.stop();
    auto spmv_time = timer.elapsed() / config.exec_iterations;
    auto gflops = A.values.size() / spmv_time / 1e6;
    printf("spmv_fff_A_load_B_load: %8.4lf ms, %8.4lf Gflops\n", spmv_time,
           gflops);
}

int main(int argc, char **argv) {
    cudaSetDevice(0);

    // XXX !!!!!! CSR format doesn't work with symmetric matrix marker format
    // !!!!!!
    CsrMatrix<int, float, device_memory> A_csr;

    Config config = program_options(argc, argv);

    if (!config.input_file.empty()) {
        read_matrix_file(A_csr, config.input_file);
    }

    printf("matrix A has shape (%d,%d) and %d entries, with density %f\n",
           A_csr.num_rows, A_csr.num_cols, A_csr.num_entries,
           static_cast<float>(A_csr.num_entries) / A_csr.num_rows /
               A_csr.num_cols);

    //? Dense matrix B

    //>>>>>>>>>>
    //>> bmp SpMV
    //>>>>>>>>>>
    CUDATimer timer;
    //? A_bmp
    BitmapCSR<int, float, bmp64_t, device_memory> A_bmp;
    thrust::device_vector<float> d_bmp_A_values;
    thrust::device_vector<int> d_bmp_A_offsets;
    timer.start();
    convert_csr2bmp(A_csr, A_bmp);
    timer.stop();
    printf("conversion time: %f ms\n", timer.elapsed());
    printf("A_bmp has shape (%d, %d) and %d entries\n", A_bmp.num_rows,
           A_bmp.num_cols, A_bmp.num_entries);
    //? A_bmp -> A_bmp
    thrust::device_vector<int> A_bmp_rowidx(A_bmp.num_rows);

    // print_vec(A_bmp.row_pointers, "A row_pointers");
    // print_vec(A_bmp.column_indices, "A column_indices");

    //? d_bmp_vec_B
    const auto nrow_aligned = A_bmp.num_rows * TILE_DIM;
    thrust::device_vector<float> B_values(nrow_aligned, 1.0);
    // convert_vec2bmp(d_vec_B, B_values);
    //? d_bmp_vec_C
    thrust::device_vector<float> C_values(nrow_aligned, 0.0);

    execute_spmv(config, A_bmp, B_values, C_values);

    // print_vec(C_values, "d_dense_C [BMP]: ");
    // printf("\n--------------shrink----------------\n");
    // ShrunkMatrix<int, float, device_memory> A_mix;
    // CPUTimer ctimer;
    // ctimer.start();

    // shrink_columns<8>(A_csr, A_mix);

    // ctimer.stop();
    // auto time_shrink = ctimer.elapsed();
    // printf("A Column shrinking time: %lf ms\n", time_shrink);
    // // report_bmp_matrix(A_mix);

    // ShrBmpCSR<int, float, bmp64_t, device_memory> A_mix_bmp;
    // timer.start();

    // convert_shr2bmp(A_mix, A_mix_bmp);

    // timer.stop();
    // auto time_conversion = timer.elapsed();
    // printf("Mix to Bitmap conversion time: %lf ms\n", time_conversion);
    // report_bmp_matrix(A_mix_bmp);

    // execute_spmv(config, A_mix_bmp, B_values, C_values);

    return 0;
}
