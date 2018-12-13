#include "gpu_helper.h"

void getInfo() {
	int device;
	checkCudaError(cudaGetDevice(&device));
	cudaDeviceProp prop;
	checkCudaError(cudaGetDeviceProperties(&prop, device));
	uint32_t numThreadsPerBlock = prop.maxThreadsPerBlock;
	uint32_t numBlocks = prop.multiProcessorCount
		* (prop.maxThreadsPerMultiProcessor / numThreadsPerBlock);
	
	cout << numThreadsPerBlock << " " << numBlocks << endl;
	cout << "memoryBusWidth: " << prop.memoryBusWidth << endl;
}