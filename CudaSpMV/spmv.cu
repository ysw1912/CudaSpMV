#include "spmv.h"

// CuSparse_SpMV封装（float型重载）
void spmvCuSparse(cusparseHandle_t &handle, int m, int n, int nnz, const float *alpha,
	cusparseMatDescr_t &descr, const float *value, const int *rowPtr, const int *col,
	const float *x, const float *beta, float *y) {
	checkCuSparseError(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, alpha, descr, value, rowPtr, col, x, beta, y));
}
// CuSparse_SpMV封装（double型重载）
void spmvCuSparse(cusparseHandle_t &handle, int m, int n, int nnz, const double *alpha,
	cusparseMatDescr_t &descr, const double *value, const int *rowPtr, const int *col,
	const double *x, const double *beta, double *y) {
	checkCuSparseError(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, alpha, descr, value, rowPtr, col, x, beta, y));
}

