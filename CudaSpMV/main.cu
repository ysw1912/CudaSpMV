#include "pagerank.h"
#include "test.cuh"

using ValueType = float;

uint32_t numThreadsPerBlock, numBlocks;

void TestSpMV(const char* file, bool isDirected, bool isWeighted, uint32_t times);
void TestPageRank(const char* file, bool isDirected, bool isWeighted, uint32_t times);

int main()
{
	//test::TestScan();
	test::TestSegmentedScan();
	return 0;

	getDeviceInfo(numThreadsPerBlock, numBlocks);

	const char* file = "cage14.mtx";
	bool isDirected = true;
	bool isWeighted = true;
	uint32_t times = 20;
	TestSpMV(file, isDirected, isWeighted, times);
	//TestPageRank(file, isDirected, isWeighted, times);
	
	return 0;
}

void TestSpMV(const char* file, bool isDirected, bool isWeighted, uint32_t times)
{
	const string FILE_PATH = string("F:\\networks\\spmv\\") + string(file);
	CooMatrix<ValueType> coo;
	if (isWeighted)
		coo.ReadFileContainValue(FILE_PATH, isDirected);
	else
		coo.ReadFileSetValue(FILE_PATH, isDirected);

	Profiler::Start();
	CsrMatrix<ValueType> csr(coo);
	Profiler::Finish();
	printf("%f (s)\nConvert to CSR format Finished.\n",
		Profiler::dumpDuration() / CLOCKS_PER_SEC);

	uint32_t N = static_cast<uint32_t>(coo.num_vertices);

	/* 可选格式 - CSR_CUSPARSE, CSR_LIGHTSPMV, BRC_SPMV, BRCP_SPMV */
	vector<ValueType> res(N);
	spmv<ValueType>(csr, &res[0], times, CSR_CUSPARSE);

	vector<ValueType> res1(N);
	spmv<ValueType>(csr, &res1[0], times, BRC_SPMV);
	vector<ValueType> res2(N);
	spmv<ValueType>(csr, &res2[0], times, BRCP_SPMV);

	ValueType re1 = RelativeError(res, res1);
	ValueType re2 = RelativeError(res, res2);
	//Func(res, res1, res2);
	printf("BRC's RE = %f, BRCP's RE = %f\n", re1, re2);
}

void TestPageRank(const char* file, bool isDirected, bool isWeighted, uint32_t times)
{
	const string FILE_PATH = string("F:\\networks\\pagerank\\") + string(file);
	CooMatrix<ValueType> coo;
	if (isWeighted)
		coo.ReadFileContainValue(FILE_PATH, isDirected);
	else
		coo.ReadFileSetValue(FILE_PATH, isDirected);

	Profiler::Start();
	initValues(coo);
	coo.transpose();
	CsrMatrix<ValueType> csr(coo);
	Profiler::Finish();
	printf("%f (s)\nConvert to CSR format Finished.\n",
		Profiler::dumpDuration() / CLOCKS_PER_SEC);

	uint32_t N = static_cast<uint32_t>(coo.num_vertices);

	/* 可选格式 - CSR_CUSPARSE, CSR_LIGHTSPMV, BRC_SPMV, BRCP_SPMV */
	PrNode<ValueType>* vecPR1 = new PrNode<ValueType>[N];
	pagerank<ValueType>(csr, vecPR1, times, CSR_LIGHTSPMV);
	PR_Print(vecPR1, N);
	
	PrNode<ValueType>* vecPR2 = new PrNode<ValueType>[N];
	pagerank<ValueType>(csr, vecPR2, times, BRC_SPMV);
	PR_Print(vecPR2, N);

	PrNode<ValueType>* vecPR3 = new PrNode<ValueType>[N];
	pagerank<ValueType>(csr, vecPR3, times, BRCP_SPMV);
	PR_Print(vecPR3, N);

	int n = 100;
	printf("BRC  前 %d 结果的相同个数: %d\n", n, VecPrComp(vecPR1, vecPR2, n));
	printf("BRCP 前 %d 结果的相同个数: %d\n", n, VecPrComp(vecPR1, vecPR3, n));
	
	delete[] vecPR1;
	delete[] vecPR2;
	delete[] vecPR3;
}