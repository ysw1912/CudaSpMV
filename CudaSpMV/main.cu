#include "pagerank.h"
#include "test.h"

using ValueType = float;

void TestSpMV();
void TestPageRank();

int main()
{
	//getInfo(); return 0;
	//test::test02();

	checkCudaError(cudaDeviceReset());
	
	//TestSpMV();
	TestPageRank();

	return 0;
}

void TestSpMV()
{
	//const string FILE_PATH = "F:\\networks\\nvidia research\\webbase-1M.mtx";
	const string FILE_PATH = "F:\\networks\\soc-sign-bitcoin-otc.mtx";
	CooMatrix<ValueType> coo;
	bool isDirected = true;
	//coo.ReadFileSetValue(FILE_PATH, isDirected);
	coo.ReadFileContainValue(FILE_PATH, isDirected);

	Profiler::Start();
	CsrMatrix<ValueType> csr(coo);
	Profiler::Finish();
	printf("%f (s)\nConvert to CSR format Finished.\n",
		Profiler::dumpDuration() / CLOCKS_PER_SEC);

	uint32_t N = static_cast<uint32_t>(coo.num_vertices);

	/* 可选格式 - CSR_CUSPARSE, CSR_LIGHTSPMV, BRC_SPMV, BRCP_SPMV */
	vector<ValueType> res(N);
	spmv<ValueType, 1>(csr, &res[0], CSR_CUSPARSE);

	vector<ValueType> res1(N);
	spmv<ValueType, 1>(csr, &res1[0], BRC_SPMV);
	vector<ValueType> res2(N);
	spmv<ValueType, 1>(csr, &res2[0], BRCP_SPMV);

	ValueType re1 = RelativeError(res, res1);
	ValueType re2 = RelativeError(res, res2);
	//Func(res, res1, res2);
	printf("BRC's RE = %f, BRCP's RE = %f\n", re1, re2);
}

void TestPageRank()
{
	//const string FILE_PATH = "F:\\networks\\real2.csv";
	const string FILE_PATH = "F:\\networks\\law\\dblp-2010.mtx";
	CooMatrix<ValueType> coo;
	bool isDirected = false;
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
	pagerank<ValueType, 1>(csr, vecPR1, CSR_CUSPARSE);
	PR_Print(vecPR1, N);
	
	PrNode<ValueType>* vecPR2 = new PrNode<ValueType>[N];
	pagerank<ValueType, 1>(csr, vecPR2, BRC_SPMV);
	PR_Print(vecPR2, N);

	PrNode<ValueType>* vecPR3 = new PrNode<ValueType>[N];
	pagerank<ValueType, 1>(csr, vecPR3, BRCP_SPMV);
	PR_Print(vecPR3, N);

	int n = 100;
	printf("BRC  前 %d 结果的相同个数: %d\n", n, VecPrComp(vecPR1, vecPR2, n));
	printf("BRCP 前 %d 结果的相同个数: %d\n", n, VecPrComp(vecPR1, vecPR3, n));
	
	delete[] vecPR1;
	delete[] vecPR2;
	delete[] vecPR3;
}