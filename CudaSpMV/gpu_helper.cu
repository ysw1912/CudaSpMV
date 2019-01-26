#include "gpu_helper.h"

void getDeviceInfo(uint32_t& numThreadsPerBlock, uint32_t& numBlocks)
{
	int device;
	checkCudaError(cudaGetDevice(&device));
	cudaDeviceProp prop;
	checkCudaError(cudaGetDeviceProperties(&prop, device));
	numThreadsPerBlock = prop.maxThreadsPerBlock;
	numBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / numThreadsPerBlock);
	printf("numThreadsPerBlock = %d, numBlocks = %d\n", numThreadsPerBlock, numBlocks);
}