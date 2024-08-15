#pragma once
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <type_traits>
#include <vector>

#include <bmp/matrices/csr.h>
#include <bmp/utils/common.h>
#include <bmp/utils/mmio.h>

namespace bmp {
// Custom exclusive scan function since C++ STL does not provide one out of the
// box
template <typename T> void exclusive_scan(T *input, int length) {
    if (length == 0 || length == 1)
        return;

    T old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++) {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

template <class CsrMatrix>
int read_from_mtx(CsrMatrix &matrix, std::string input) {
    using IndexType = typename CsrMatrix::index_type;
    using ValueType = typename CsrMatrix::value_type;

    int nrow, ncol;
    IndexType nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    IndexType nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0,
        isComplex = 0;
    // load matrix
    char *filename = const_cast<char *>(input.c_str());

    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode)) {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode)) {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode)) {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode)) {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &nrow, &ncol, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
        isSymmetric_tmp = 1;
        // printf("input matrix is symmetric = true\n");
    } else {
        // printf("input matrix is symmetric = false\n");
    }

    thrust::host_vector<IndexType> csrRowPtr_counter(nrow + 1);
    thrust::host_vector<IndexType> csrRowIdx_tmp(nnz_mtx_report);
    thrust::host_vector<IndexType> csrColIdx_tmp(nnz_mtx_report);
    thrust::host_vector<ValueType> csrVal_tmp(nnz_mtx_report);

    // int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    // int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    // ValueType *csrVal_tmp =
    //     (ValueType *)malloc(nnz_mtx_report * sizeof(ValueType));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (IndexType i = 0; i < nnz_mtx_report; i++) {
        int idxi, idxj;
        double fval, fval_im;
        int ival;

        if (isReal) {
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        } else if (isComplex) {
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        } else if (isInteger) {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        } else if (isPattern) {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp) {
        for (IndexType i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter.data(), nrow + 1);

    nnz_tmp = csrRowPtr_counter[nrow];

    thrust::host_vector<IndexType> csrRowPtr_alias = csrRowPtr_counter;
    thrust::host_vector<IndexType> csrColIdx_alias(nnz_tmp);
    thrust::host_vector<ValueType> csrVal_alias(nnz_tmp);

    std::fill(csrRowPtr_counter.begin(), csrRowPtr_counter.end(), 0);

    if (isSymmetric_tmp) {
        for (IndexType i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
                IndexType offset = csrRowPtr_alias[csrRowIdx_tmp[i]] +
                                   csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] +
                         csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            } else {
                IndexType offset = csrRowPtr_alias[csrRowIdx_tmp[i]] +
                                   csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    } else {
        for (IndexType i = 0; i < nnz_mtx_report; i++) {
            IndexType offset = csrRowPtr_alias[csrRowIdx_tmp[i]] +
                               csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    matrix.num_rows = nrow;
    matrix.num_cols = ncol;
    matrix.num_entries = nnz_tmp;

    matrix.row_pointers = csrRowPtr_alias;
    matrix.column_indices = csrColIdx_alias;
    matrix.values = csrVal_alias;

    return 0;
}

bool inline string_end_with(std::string base, std::string postfix) {
    std::regex string_end(".*" + postfix + "$");
    std::smatch base_match;
    return std::regex_match(base, base_match, string_end);
}

template <class CsrMatrix>
void read_from_csr(CsrMatrix &matrix, const std::string &filename) {
    std::ifstream csr_file;
    csr_file.open(filename, std::ios::binary);
    if (!csr_file.is_open()) {
        std::cout << "cannot open csr file!" << std::endl;
        std::exit(1);
    }
    int nrow;
    int nnz;
    csr_file.read(reinterpret_cast<char *>(&nrow), sizeof(int));
    csr_file.read(reinterpret_cast<char *>(&nnz), sizeof(int));

    thrust::host_vector<int> row_ptr(nrow + 1);
    thrust::host_vector<int> col_idx(nnz);

    csr_file.read(reinterpret_cast<char *>(row_ptr.data()),
                  (nrow + 1) * sizeof(int));
    csr_file.read(reinterpret_cast<char *>(col_idx.data()), nnz * sizeof(int));

    assert(row_ptr[nrow] == nnz);
    //    row_pointers[nrow] = nnz;

    csr_file.close();
    matrix.resize(nrow, nrow, nnz);
    matrix.row_pointers = row_ptr;
    matrix.column_indices = col_idx;
    thrust::fill(matrix.values.begin(), matrix.values.end(), 1.0);
}

template <class CsrMatrix>
void read_matrix_file(CsrMatrix &d_csr_A, std::string input) {
    if (string_end_with(input, ".mtx")) {
        read_from_mtx(d_csr_A, input);
        // DCsr A;
        // read_from_mtx_cusp(A, input);
        // compare_csr(d_csr_A, A);
    } else if (string_end_with(input, ".csr")) {
        read_from_csr(d_csr_A, input);
    } else {
        printf("file format is not supported\n");
        std::exit(1);
    }
}

}  // namespace bmp