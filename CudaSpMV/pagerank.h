#ifndef __PAGERANK__
#define __PAGERANK__

#include "spmv.h"

extern uint32_t numThreadsPerBlock, numBlocks;

template <typename ValueType>
struct PrNode 
{
	size_t index;
	ValueType score;
	
	static bool greater(const PrNode& lhs, const PrNode& rhs)
	{
		return lhs.score > rhs.score;
	}
};

// 输出PageRank结果向量
template <class ValueType>
void PR_Print(const PrNode<ValueType>* pr_vector, size_t size, size_t top = 20)
{
	size = MIN(size, top);
	printf("*************************************\n");
	printf("******PageRank排序结果（前%zd）******\n", size);
	cout << setiosflags(ios::left) << setw(10) << "排名";
	cout << setiosflags(ios::left) << setw(14) << "节点编号";
	cout << setiosflags(ios::left) << setw(14) << "PageRank分值" << endl;
	for (size_t i = 0; i < size; i++) {
		cout << setiosflags(ios::left) << setw(10) << i + 1;
		cout << setiosflags(ios::left) << setw(14) << pr_vector[i].index;
		cout << setiosflags(ios::left) << setw(14) << pr_vector[i].score << endl;
	}
	printf("*************************************\n\n");
}

template <class ValueType>
int VecPrComp(const PrNode<ValueType>* VecPR1,
			  const PrNode<ValueType>* VecPR2,
			  int n = 100)
{
	int count = 0;
	for (size_t i = 0; i < n; ++i) {
		if (VecPR1[i].index == VecPR2[i].index)
			++count;
	}
	return count;
}

// COO矩阵生成values
template <class ValueType>
void initValues(CooMatrix<ValueType>& coo)
{
	for (size_t j = 0; j < coo.num_edges; ) {
		size_t i = j;
		while (j < coo.num_edges && coo.row[j] == coo.row[i])	++j;
		for (size_t k = i; k < j; ++k) {
			coo.value[k] = (ValueType)1.0 / (j - i);
		}
	}
}

// CSR矩阵生成values
template <class ValueType>
void initValues(CsrMatrix<ValueType>& csr)
{
	for (size_t i = 0; i < csr.num_vertices; ++i) {
		uint32_t div = csr.rowPtr[i + 1] - csr.rowPtr[i];
		for (uint32_t j = csr.rowPtr[i]; j < csr.rowPtr[i + 1]; ++j) {
			csr.value[j] = static_cast<ValueType>(1.0) / static_cast<ValueType>(div);
		}
	}
}

/* ---------------------------- CSR_CuSparse计算PageRank ---------------------------- */
// CuSparse库计算PageRank
template <class ValueType>
void pagerankCuSparse(CsrMatrix<ValueType>& csr,
					  PrNode<ValueType>*& vecPR,
					  float& total_time,
					  const ValueType damping,
	                  const ValueType tolerant)
{
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
	ValueType *d_value, *d_x, *d_y, *d_error;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_value, num_edges * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
	checkCudaError(cudaMalloc((void**)&d_rowPtr, (num_vertices + 1) * sizeof(int)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_error, sizeof(ValueType)));

	checkCudaError(cudaMemcpy(d_value, &csr.value[0], num_edges * sizeof(ValueType), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, &csr.col[0], num_edges * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_rowPtr, &csr.rowPtr[0], (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));

	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0 / num_vertices);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	Profiler::Finish();
	cout << Profiler::dumpDuration() / CLOCKS_PER_SEC << " (s) " << endl;

	GpuTimer timer;
	timer.Start();
	cout << "CSR-CuSparse PageRank Start..." << endl;

	ValueType error = -1.0;
	uint16_t count = 1;
	uint32_t rep = (uint32_t)ceil((float)num_vertices / (numThreadsPerBlock * numBlocks));

	do {
		ValueType a = ValueType(1), b = ValueType(0);
		spmvCuSparse(handle, num_vertices, num_vertices, num_edges, &a, descr, d_value, d_rowPtr, d_col, d_x, &b, d_y);
		PR_Update<<<numBlocks, numThreadsPerBlock>>>(damping, num_vertices, rep, d_x, d_y);
		checkCudaError(cudaMemset(d_error, 0, sizeof(ValueType)));
		/*
		thrust::device_ptr<ValueType> dev_y(d_y);
		error = thrust::reduce(dev_y, dev_y + num_vertices, ValueType(0), thrust::plus<ValueType>());
		*/
		Reduce<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(ValueType) >> > (d_y, num_vertices, d_error);
		checkCudaError(cudaMemcpy(&error, d_error, sizeof(ValueType), cudaMemcpyDeviceToHost));
		//printf("第%d%d轮\t\tError: %f\n", count / 10, count % 10, error);
		++count;
	} while (error > tolerant);

	timer.Stop();
	printf("CSR-CuSparse PageRank Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_rowPtr));
	checkCudaError(cudaFree(d_error));
	checkCudaError(cudaFree(d_y));

	ValueType *x = new ValueType[num_vertices];
	checkCudaError(cudaMemcpy(x, d_x, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_x));

	checkCuSparseError(cusparseDestroyMatDescr(descr));
	checkCuSparseError(cusparseDestroy(handle));

	for (size_t i = 0; i < num_vertices; i++) {
		vecPR[i].index = i;
		vecPR[i].score = x[i];
	}
	delete[] x;
}
/* ------------------------------------------------------------------------------- */

/* ---------------------------- LightSpMV计算PageRank ---------------------------- */

// 设置纹理对象属性（float型重载）
void setResDesc(cudaResourceDesc &resDesc, float *d_error);
// 设置纹理对象属性（double型重载）
void setResDesc(cudaResourceDesc &resDesc, double *d_error);

// LightSpMV计算PageRank
template <class ValueType>
void pagerankLightSpMV(CsrMatrix<ValueType>& csr, 
					   PrNode<ValueType>*& vecPR,
					   float& total_time,
					   const ValueType damping,
					   const ValueType tolerant)
{
	Profiler::Start();

	uint32_t num_vertices = (uint32_t)csr.num_vertices, num_edges = (uint32_t)csr.num_edges;

	uint32_t *d_col, *d_rowPtr, *d_rowCounter;
	ValueType *d_value, *d_x, *d_y, *d_error;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_value, num_edges * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_col, num_edges * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_rowPtr, (num_vertices + 1) * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_error, sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_rowCounter, sizeof(uint32_t)));

	checkCudaError(cudaMemcpy(d_value, &csr.value[0], num_edges * sizeof(ValueType), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, &csr.col[0], num_edges * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_rowPtr, &csr.rowPtr[0], (num_vertices + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0 / num_vertices);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	///*  bbs.csdn.net/topics/390886235
	//blog.csdn.net/kelvin_yan/article/details/54019017
	//blog.csdn.net/yanghangjun/article/details/5588284
	//*/
	///* 纹理对象（CC3.0+） */
	//cudaTextureDesc texDesc;				// 纹理描述符，用来描述纹理参数
	//cudaResourceDesc resDesc;				// 资源描述符，用来获取纹理数据
	//cudaTextureObject_t texVectorX = 0;		// 需要创建的纹理对象

	//memset(&texDesc, 0, sizeof(texDesc));
	//texDesc.addressMode[0] = cudaAddressModeClamp;	// 寻址模式，默认
	//texDesc.addressMode[1] = cudaAddressModeClamp;
	//texDesc.filterMode = cudaFilterModePoint;		// 滤波模式，返回相应坐标下，最接近的元素值
	//texDesc.readMode = cudaReadModeElementType;		// 读取模式，返回回原始类型，不做转换

	//memset(&resDesc, 0, sizeof(resDesc));
	//resDesc.resType = cudaResourceTypeLinear;	// 一维数组
	//setResDesc(resDesc, d_error);	// texels的属性
	//resDesc.res.linear.devPtr = (void *)thrust::raw_pointer_cast(dev_x); // 设备指针
	//resDesc.res.linear.sizeInBytes = num_vertices * sizeof(ValueType);	// 数组的字节长度
	///* 创建纹理对象 */
	//checkCudaError(cudaCreateTextureObject(&texVectorX, &resDesc, &texDesc, NULL));

	Profiler::Finish();
	cout << Profiler::dumpDuration() / CLOCKS_PER_SEC << " (s) " << endl;

	GpuTimer timer;
	timer.Start();
	cout << "CSR-LightSpMV PageRank Start..." << endl;

	ValueType error;
	uint16_t count = 1;
	uint32_t rep = (uint32_t)ceil((float)num_vertices / (numThreadsPerBlock * numBlocks));
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
		PR_Update<<<numBlocks, numThreadsPerBlock>>>(damping, num_vertices, rep, d_x, d_y);
		checkCudaError(cudaMemset(d_error, 0, sizeof(ValueType)));
		Reduce<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(ValueType)>>>(d_y, num_vertices, d_error);
		checkCudaError(cudaMemcpy(&error, d_error, sizeof(ValueType), cudaMemcpyDeviceToHost));
		//printf("第%d%d轮\t\tError: %f\n", count / 10, count % 10, error);
		count++;
	} while (error > tolerant);

	timer.Stop();
	printf("CSR-LightSpMV PageRank Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_rowPtr));
	checkCudaError(cudaFree(d_y));
	checkCudaError(cudaFree(d_error));
	checkCudaError(cudaFree(d_rowCounter));

	ValueType *x = new ValueType[num_vertices];
	checkCudaError(cudaMemcpy(x, d_x, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_x));

	for (size_t i = 0; i < num_vertices; i++) {
		vecPR[i].index = i;
		vecPR[i].score = x[i];
	}
	delete[] x;
}
/* ------------------------------------------------------------------------------- */

/* ----------------------------- BRCSpMV计算PageRank ----------------------------- */
// BrcSpMV计算PageRank
template <typename ValueType>
void pagerankBrcSpMV(BrcMatrix<ValueType>& brc,
					 PrNode<ValueType>*& vecPR,
					 float& total_time,
					 const ValueType damping,
					 const ValueType tolerant)
{
	Profiler::Start();

	uint32_t num_vertices = (uint32_t)brc.num_vertices;
	uint32_t *d_rowPerm, *d_col, *d_blockPtr, *d_block_width, *d_numBlocks;
	ValueType *d_value, *d_x, *d_y, *d_error;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_rowPerm, brc.rowPerm.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_col, brc.col.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_blockPtr, brc.blockPtr.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_block_width, brc.block_width.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_numBlocks, sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_value, brc.value.size() * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_error, sizeof(ValueType)));

	checkCudaError(cudaMemcpy(d_rowPerm, &brc.rowPerm[0], brc.rowPerm.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, &brc.col[0], brc.col.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_blockPtr, &brc.blockPtr[0], brc.blockPtr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_block_width, &brc.block_width[0], brc.block_width.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_numBlocks, &brc.numBlocks, sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_value, &brc.value[0], brc.value.size() * sizeof(ValueType), cudaMemcpyHostToDevice));

	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0 / num_vertices);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	Profiler::Finish();
	cout << Profiler::dumpDuration() / CLOCKS_PER_SEC << " (s) " << endl;

	GpuTimer timer;
	timer.Start();
	cout << "BRC-SpMV PageRank Start..." << endl;
	ValueType error;
	uint32_t count = 1;
	do {
		uint32_t rep = (uint32_t)ceil((float)brc.numBlocks * 32 / (numThreadsPerBlock * numBlocks));
		brcspmv::brcSpMV<ValueType><<<numBlocks, numThreadsPerBlock>>>(rep, 32, d_rowPerm, d_col, d_value, d_blockPtr, d_block_width, d_numBlocks, d_x, d_y);
		rep = (uint32_t)ceil((float)num_vertices / (numThreadsPerBlock * numBlocks));
		PR_Update<<<numBlocks, numThreadsPerBlock>>>(damping, num_vertices, rep, d_x, d_y);
		checkCudaError(cudaMemset(d_error, 0, sizeof(ValueType)));
		Reduce<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(ValueType)>>>(d_y, num_vertices, d_error);
		checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));
		checkCudaError(cudaMemcpy(&error, d_error, sizeof(ValueType), cudaMemcpyDeviceToHost));
		//printf("第%d%d轮\t\tError: %f\n", count / 10, count % 10, error);
		count++;
	} while (error > tolerant);

	timer.Stop();
	printf("BRC-SpMV PageRank Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_rowPerm));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_blockPtr));
	checkCudaError(cudaFree(d_block_width));
	checkCudaError(cudaFree(d_numBlocks));
	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_y));
	checkCudaError(cudaFree(d_error));

	ValueType *x = new ValueType[num_vertices];
	checkCudaError(cudaMemcpy(x, d_x, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_x));

	for (size_t i = 0; i < num_vertices; i++) {
		vecPR[i].index = i;
		vecPR[i].score = x[i];
	}
	delete[] x;
}
/* ------------------------------------------------------------------------------- */

/* ----------------------------- BrcPSpMV计算PageRank ----------------------------- */
// BrcPSpMV计算PageRank
template <typename ValueType>
void pagerankBrcPSpMV(BrcPMatrix<ValueType>& brcP,
					  PrNode<ValueType>*& vecPR,
					  float& total_time,
					  const ValueType damping,
					  const ValueType tolerant)
{
	Profiler::Start();

	uint32_t num_vertices = static_cast<uint32_t>(brcP.num_vertices);
	uint32_t num_blocks = static_cast<uint32_t>(brcP.blockPtr.size()) - 1;
	uint32_t *d_rowPerm, *d_rowSegLen, *d_col, *d_blockPtr;
	ValueType *d_value, *d_x, *d_y, *d_error;

	printf("数据传输...\t\t");
	checkCudaError(cudaMalloc((void**)&d_rowPerm, brcP.rowPerm.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_rowSegLen, brcP.rowSegLen.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_col, brcP.col.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_value, brcP.value.size() * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_blockPtr, brcP.blockPtr.size() * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_x, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_y, num_vertices * sizeof(ValueType)));
	checkCudaError(cudaMalloc((void**)&d_error, sizeof(ValueType)));

	checkCudaError(cudaMemcpy(d_rowPerm, brcP.rowPerm.data(), brcP.rowPerm.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_rowSegLen, brcP.rowSegLen.data(), brcP.rowSegLen.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_col, brcP.col.data(), brcP.col.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_value, brcP.value.data(), brcP.value.size() * sizeof(ValueType), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_blockPtr, brcP.blockPtr.data(), brcP.blockPtr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	thrust::device_ptr<ValueType> dev_x(d_x);
	thrust::fill(dev_x, dev_x + num_vertices, (ValueType)1.0 / num_vertices);
	checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));

	Profiler::Finish();
	cout << Profiler::dumpDuration() / CLOCKS_PER_SEC << " (s) " << endl;

	GpuTimer timer;
	timer.Start();
	cout << "BRCP-SpMV PageRank Start..." << endl;
	ValueType error;
	uint32_t count = 1;
	uint32_t B1 = 32;
	uint32_t repSpMV = (uint32_t)ceil((float)num_blocks * B1 / (numThreadsPerBlock * numBlocks));
	uint32_t repPR = (uint32_t)ceil((float)num_vertices / (numThreadsPerBlock * numBlocks));
	do {
		brcspmv::brcPlusSpMV<ValueType><<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(ValueType)>>>(repSpMV, B1, num_blocks, d_rowPerm, d_rowSegLen, d_col, d_value, d_blockPtr, d_x, d_y);
		PR_Update<<<numBlocks, numThreadsPerBlock>>>(damping, num_vertices, repPR, d_x, d_y);
		checkCudaError(cudaMemset(d_error, 0, sizeof(ValueType)));
		Reduce<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(ValueType)>>>(d_y, num_vertices, d_error);
		checkCudaError(cudaMemset(d_y, 0, num_vertices * sizeof(ValueType)));
		checkCudaError(cudaMemcpy(&error, d_error, sizeof(ValueType), cudaMemcpyDeviceToHost));
		//printf("第%d%d轮\t\tError: %f\n", count / 10, count % 10, error);
		count++;
	} while (error > tolerant);

	timer.Stop();
	printf("BRCP-SpMV PageRank Time: %f (ms)\n", timer.Elapsed());
	total_time += timer.Elapsed();

	checkCudaError(cudaFree(d_rowPerm));
	checkCudaError(cudaFree(d_col));
	checkCudaError(cudaFree(d_value));
	checkCudaError(cudaFree(d_blockPtr));
	checkCudaError(cudaFree(d_y));
	checkCudaError(cudaFree(d_error));

	ValueType *x = new ValueType[num_vertices];
	checkCudaError(cudaMemcpy(x, d_x, num_vertices * sizeof(ValueType), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_x));

	for (size_t i = 0; i < num_vertices; i++) {
		vecPR[i].index = i;
		vecPR[i].score = x[i];
	}
	delete[] x;
}
/* ------------------------------------------------------------------------------- */

// pagerank
template <class ValueType>
void pagerank(CsrMatrix<ValueType> &csr,
			  PrNode<ValueType>*& vecPR,
			  uint32_t times,
			  SpMV_Method sm, 
			  const ValueType damping = 0.85,
			  const ValueType tolerant = 1.0e-5)
{
	checkCudaError(cudaDeviceReset());

	float total_time = 0.0;

	switch (sm)
	{
	case CSR_CUSPARSE: {
		for (uint32_t i = 0; i < times; ++i)
			pagerankCuSparse(csr, vecPR, total_time, damping, tolerant);
		printf("Average CSR-CuSparse PageRank Time: ");
		break;
	}
	case CSR_LIGHTSPMV: {
		for (uint32_t i = 0; i < times; ++i)
			pagerankLightSpMV<ValueType>(csr, vecPR, total_time, damping, tolerant);
		printf("Average CSR-LightSpMV PageRank Time: ");
		break;
	}
	case BRC_SPMV: {
		Profiler::Start();
		BrcMatrix<ValueType> brc(csr);
		Profiler::Finish();
		printf("%f (s)\nConvert to BRC format Finished.\n",
			Profiler::dumpDuration() / CLOCKS_PER_SEC);
		for (uint32_t i = 0; i < times; ++i)
			pagerankBrcSpMV<ValueType>(brc, vecPR, total_time, damping, tolerant);
		printf("Average BRC-SpMV PageRank Time: ");
		break;
	}
	case BRCP_SPMV: {
		Profiler::Start();
		BrcPMatrix<ValueType> brcP(csr);
		Profiler::Finish();
		printf("%f (s)\nConvert to BRCP format Finished.\n",
			Profiler::dumpDuration() / CLOCKS_PER_SEC);
		for (uint32_t i = 0; i < times; ++i)
			pagerankBrcPSpMV<ValueType>(brcP, vecPR, total_time, damping, tolerant);
		printf("Average BRCP-SpMV PageRank Time: ");
		break;
	}
	default:
		break;
	}
	printf("%f (ms)\n\n", total_time / times);

	// PR结果排序
	std::stable_sort(vecPR, vecPR + csr.num_vertices, PrNode<ValueType>::greater);
}

// 更新PR向量x，将误差存于y
template <typename ValueType>
__global__ void PR_Update(ValueType alpha, size_t num_vertices, uint32_t rep,
	ValueType *x, ValueType *y) {
	uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;
	for (uint32_t i = 0; i < rep; i++) {
		if (row < (uint32_t)num_vertices) {
			ValueType last_result = x[row];
			x[row] = alpha * y[row] + (1.0 - alpha) / num_vertices;
			y[row] = x[row] > last_result ? x[row] - last_result : last_result - x[row];
		}
		row += gridDim.x * blockDim.x;
	}
}

#endif