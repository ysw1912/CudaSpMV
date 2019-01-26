#ifndef __SPMV__
#define __SPMV__

#include "brc_spmv_core.h"
#include "light_spmv_core.h"
#include "sparse_matrix.h"

extern uint32_t numThreadsPerBlock, numBlocks;

enum SpMV_Method
{
	CSR_CUSPARSE, CSR_LIGHTSPMV, BRC_SPMV, BRCP_SPMV, NUM_SPMV_METHODS
};

/* ---------------------------- CSR_CuSparse计算SpMV ---------------------------- */

// CuSparse_SpMV封装（float型重载）
void spmvCuSparse(cusparseHandle_t &handle, int m, int n, int nnz, const float *alpha,
	cusparseMatDescr_t &descr, const float *value, const int *rowPtr, const int *col,
	const float *x, const float *beta, float *y);
// CuSparse_SpMV封装（double型重载）
void spmvCuSparse(cusparseHandle_t &handle, int m, int n, int nnz, const double *alpha,
	cusparseMatDescr_t &descr, const double *value, const int *rowPtr, const int *col,
	const double *x, const double *beta, double *y);

// CuSparse库计算PageRank
template <class ValueType>
void spmvCuSparse(CsrMatrix<ValueType>& csr, uint32_t N, ValueType* res, float& total_time)
{
	checkCudaError(cudaDeviceReset());
	Profiler::Start();

	// 创建cuSPARSE handle
	cusparseHandle_t handle = 0;
	checkCuSparseError(cusparseCreate(&handle));

	// 构造矩阵的descriptor
	cusparseMatDescr_t descr = 0;
	checkCuSparseError(cusparseCreateMatDescr(&descr));
	checkCuSparseError(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCuSparseError(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

	int num_vertices = int(csr.num_vertices), num_edges = int(csr.num_edges);

	int *d_col, *d_rowPtr;
	ValueType *d_value, *d_x, *d_y;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_value, num_edges * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
	checkCudaError(cudaMalloc((void**)&d_rowPtr, (num_vertices + 1) * sizeof(int)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));

	checkCudaError(cudaMemcpy(d_rowPtr, csr.rowPtr.data(), (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_value, csr.value.data(), num_edges * sizeof(ValueType), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, csr.col.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
	
	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	Profiler::Finish();
	printf("%f (s)\n", Profiler::dumpDuration() / CLOCKS_PER_SEC);

	GpuTimer timer;
	timer.Start();
	printf("CSR-CuSparse SpMV Start...\n");

	uint16_t count = 1;
	do {
		ValueType a = ValueType(1), b = ValueType(0);
		spmvCuSparse(handle, num_vertices, num_vertices, num_edges, &a, descr, d_value, d_rowPtr, d_col, d_x, &b, d_y);
		checkCudaError(cudaMemcpy(d_x, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToDevice));
		printf("第%d%d轮\n", count / 10, count % 10);
		++count;
	} while (count <= N);

	timer.Stop();
	printf("CSR-CuSparse SpMV Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_rowPtr));
	checkCudaError(cudaFree(d_x));

	checkCudaError(cudaMemcpy(res, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_y));

	checkCuSparseError(cusparseDestroyMatDescr(descr));
	checkCuSparseError(cusparseDestroy(handle));
}

/* ---------------------------- LightSpMV计算SpMV ---------------------------- */

// LightSpMV封装（float型重载）
template <typename XType, uint32_t THREADS_PER_VECTOR, uint32_t VECTORS_PER_BLOCK>
void spmvLight(uint32_t *d_rowCounter, const uint32_t num_vertices,
	const uint32_t *d_rowPtr, const uint32_t *d_col, const float *d_value,
	XType vectorX, float *d_y, const float alpha, const float beta) {
	lightspmv::csr32DynamicWarpBLAS<float, XType, THREADS_PER_VECTOR, VECTORS_PER_BLOCK> << <4, 1024 >> > (
		d_rowCounter, num_vertices, d_rowPtr, d_col, d_value, vectorX, d_y, alpha, beta);
}
// LightSpMV封装（double型重载）
template <typename XType, uint32_t THREADS_PER_VECTOR, uint32_t VECTORS_PER_BLOCK>
void spmvLight(uint32_t *d_rowCounter, const uint32_t num_vertices,
	const uint32_t *d_rowPtr, const uint32_t *d_col, const double *d_value,
	XType vectorX, double *d_y, const double alpha, const double beta) {
	lightspmv::csr64DynamicWarpBLAS<double, XType, THREADS_PER_VECTOR, VECTORS_PER_BLOCK> << <4, 1024 >> > (
		d_rowCounter, num_vertices, d_rowPtr, d_col, d_value, vectorX, d_y, alpha, beta);
}

// LightSpMV计算SpMV
template <class ValueType>
void spmvLight(CsrMatrix<ValueType>& csr, uint32_t N, ValueType* res, float& total_time)
{
	checkCudaError(cudaDeviceReset());
	Profiler::Start();

	uint32_t num_vertices = (uint32_t)csr.num_vertices, num_edges = (uint32_t)csr.num_edges;

	uint32_t *d_col, *d_rowPtr, *d_rowCounter;
	ValueType *d_value, *d_x, *d_y;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_value, num_edges * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_col, num_edges * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_rowPtr, (num_vertices + 1) * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_rowCounter, sizeof(uint32_t)));

	checkCudaError(cudaMemcpy(d_value, csr.value.data(), num_edges * sizeof(ValueType), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, csr.col.data(), num_edges * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_rowPtr, csr.rowPtr.data(), (num_vertices + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	Profiler::Finish();
	printf("%f (s)\n", Profiler::dumpDuration() / CLOCKS_PER_SEC);

	GpuTimer timer;
	timer.Start();
	printf("CSR-LightSpMV PageRank Start...\n");

	uint16_t count = 1;
	uint32_t mean_nnz = (uint32_t)rint((float)num_edges / num_vertices);
	do {
		ValueType alpha = ValueType(1), beta = ValueType(0);
		checkCudaError(cudaMemset(d_rowCounter, 0, sizeof(uint32_t)));
		/* 启动LightSpMV Kernel */
		if (mean_nnz <= 2)
			spmvLight<ValueType*, 2, 1024 / 2>(d_rowCounter, num_vertices, d_rowPtr, d_col, d_value, d_x, d_y, alpha, beta);
		else if (mean_nnz <= 4)
			spmvLight<ValueType*, 4, 1024 / 4>(d_rowCounter, num_vertices, d_rowPtr, d_col, d_value, d_x, d_y, alpha, beta);
		else if (mean_nnz <= 64)
			spmvLight<ValueType*, 8, 1024 / 8>(d_rowCounter, num_vertices, d_rowPtr, d_col, d_value, d_x, d_y, alpha, beta);
		else
			spmvLight<ValueType*, 32, 1024 / 32>(d_rowCounter, num_vertices, d_rowPtr, d_col, d_value, d_x, d_y, alpha, beta);
		checkCudaError(cudaMemcpy(d_x, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToDevice));
		printf("第%d%d轮\n", count / 10, count % 10);
		count++;
	} while (count <= N);

	timer.Stop();
	printf("CSR-LightSpMV SpMV Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_rowPtr));
	checkCudaError(cudaFree(d_x));
	checkCudaError(cudaFree(d_rowCounter));

	checkCudaError(cudaMemcpy(res, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_y));
}

// BRC-based SpMV
template <typename ValueType>
void spmvBrc(BrcMatrix<ValueType>& brc, uint32_t N, ValueType* res, float& total_time)
{
	checkCudaError(cudaDeviceReset());
	Profiler::Start();

	uint32_t num_vertices = (uint32_t)brc.num_vertices;
	uint32_t *d_rowPerm, *d_col, *d_blockPtr, *d_block_width, *d_numBlocks;
	ValueType *d_value, *d_x, *d_y;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_rowPerm, brc.rowPerm.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_col, brc.col.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_blockPtr, brc.blockPtr.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_block_width, brc.block_width.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_numBlocks, sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_value, brc.value.size() * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));

	checkCudaError(cudaMemcpy(d_rowPerm, brc.rowPerm.data(), brc.rowPerm.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, brc.col.data(), brc.col.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_blockPtr, brc.blockPtr.data(), brc.blockPtr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_block_width, brc.block_width.data(), brc.block_width.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_numBlocks, &brc.numBlocks, sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_value, brc.value.data(), brc.value.size() * sizeof(ValueType), cudaMemcpyHostToDevice));

	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	Profiler::Finish();
	printf("%f (s)\n", Profiler::dumpDuration() / CLOCKS_PER_SEC);

	GpuTimer timer;
	timer.Start();
	printf("BRC-based SpMV Start...\n");
	
	uint32_t count = 1;
	uint32_t rep = (uint32_t)ceil((float)brc.numBlocks * 32 / (numBlocks * numThreadsPerBlock));
	do {
		brcspmv::brcSpMV<ValueType><<<numBlocks, numThreadsPerBlock>>>(rep, 32, d_rowPerm, d_col, d_value, d_blockPtr, d_block_width, d_numBlocks, d_x, d_y);
		checkCudaError(cudaMemcpy(d_x, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToDevice));
		printf("第%d%d轮\n", count / 10, count % 10);
		count++;
	} while (count <= N);

	timer.Stop();
	printf("BRC-based SpMV Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_rowPerm));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_blockPtr));
	checkCudaError(cudaFree(d_block_width));
	checkCudaError(cudaFree(d_numBlocks));
	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_x));

	checkCudaError(cudaMemcpy(res, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_y));
}

// BRCP-based SpMV
template <typename ValueType>
void spmvBrcp(BrcPMatrix<ValueType>& brcP, uint32_t N, ValueType* res, float& total_time)
{
	checkCudaError(cudaDeviceReset());
	Profiler::Start();

	uint32_t num_vertices = static_cast<uint32_t>(brcP.num_vertices);
	uint32_t num_blocks = static_cast<uint32_t>(brcP.blockPtr.size()) - 1;
	uint32_t *d_rowPerm, *d_rowSegLen, *d_col, *d_blockPtr;
	ValueType *d_value, *d_x, *d_y;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_rowPerm, brcP.rowPerm.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_rowSegLen, brcP.rowSegLen.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_col, brcP.col.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_value, brcP.value.size() * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_blockPtr, brcP.blockPtr.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));

	checkCudaError(cudaMemcpy(d_rowPerm, brcP.rowPerm.data(), brcP.rowPerm.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_rowSegLen, brcP.rowSegLen.data(), brcP.rowSegLen.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, brcP.col.data(), brcP.col.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_value, brcP.value.data(), brcP.value.size() * sizeof(ValueType), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_blockPtr, brcP.blockPtr.data(), brcP.blockPtr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	Profiler::Finish();
	printf("%f (s)\n", Profiler::dumpDuration() / CLOCKS_PER_SEC);

	GpuTimer timer;
	timer.Start();
	printf("BRCP-based SpMV Start...\n");
	uint32_t count = 1;
	uint32_t B1 = 32;
	uint32_t rep = (uint32_t)ceil((float)num_blocks * B1 / (numBlocks * numThreadsPerBlock));
	do {
		brcspmv::brcPlusSpMV<ValueType><<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(ValueType)>>>(rep, B1, num_blocks, d_rowPerm, d_rowSegLen, d_col, d_value, d_blockPtr, d_x, d_y);
		checkCudaError(cudaMemcpy(d_x, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToDevice));
		printf("第%d%d轮\n", count / 10, count % 10);
		count++;
	} while (count <= N);

	timer.Stop();
	printf("BRCP-based SpMV Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_rowPerm));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_blockPtr));
	checkCudaError(cudaFree(d_x));

	checkCudaError(cudaMemcpy(res, d_y, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_y));
}

/* ---------------------------- SpMV ---------------------------- */

template <class ValueType, uint32_t ITER>
void spmv(CsrMatrix<ValueType> &csr, ValueType* res, SpMV_Method sm)
{
	checkCudaError(cudaDeviceReset());

	float total_time = 0.0;

	switch (sm)
	{
	case CSR_CUSPARSE: {
		spmvCuSparse(csr, ITER, res, total_time);
		printf("Average CSR-based CuSparseSpMV Time: ");
		break;
	}
	case CSR_LIGHTSPMV: {
		spmvLight<ValueType>(csr, ITER, res, total_time);
		printf("Average CSR-based LightSpMV Time: ");
		break;
	}
	case BRC_SPMV: {
		Profiler::Start();
		BrcMatrix<ValueType> brc(csr);
		Profiler::Finish();
		printf("%f (s)\nConvert to BRC format Finished.\n",
			Profiler::dumpDuration() / CLOCKS_PER_SEC);
		spmvBrc<ValueType>(brc, ITER, res, total_time);
		printf("Average BRC-based SpMV Time: ");
		break;
	}
	case BRCP_SPMV: {
		Profiler::Start();
		BrcPMatrix<ValueType> brcP(csr);
		Profiler::Finish();
		printf("%f (s)\nConvert to BRCP format Finished.\n",
			Profiler::dumpDuration() / CLOCKS_PER_SEC);
		spmvBrcp<ValueType>(brcP, ITER, res, total_time);
		cout << "Average BRCP-based SpMV Time: ";
		break;
	}
	default:
		break;
	}
	printf("%f (ms)\n\n", total_time / ITER);
}

#endif