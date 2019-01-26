#ifndef __GPU_HELPER__
#define __GPU_HELPER__

#include "cpu_helper.h"
#include "gputimer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include <sm_20_intrinsics.h>
#include <sm_30_intrinsics.h>	// for __syncwarp()

/* CuSparse */
#include <cusparse.h>

/* Thrust */
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\fill.h>
//#include <thrust\reduce.h>
//#include <thrust\sort.h>

#include <cooperative_groups.h>

#if defined(__cplusplus) && defined(__CUDACC__)
# include "cooperative_groups_helpers.h"
namespace cg = cooperative_groups;
#endif

#define checkCudaError(err) __CheckCudaError( err, __FILE__, __LINE__ )
static void __CheckCudaError(cudaError_t err, const char *file, const int32_t line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(-1);
	}
}

#define checkCuSparseError(status) __CheckCusparseError( status, __FILE__, __LINE__ )
static void __CheckCusparseError(cusparseStatus_t status, const char *file, const int32_t line)
{
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CusparseError in %s at line %d\n", file, line);
		exit(-1);
	}
}

// 获取GPU参数信息
void getDeviceInfo(uint32_t& numThreadsPerBlock, uint32_t& numBlocks);

// 大小未定的共享内存工具类
// 使用extern避免链接错误
template <class T>
struct SharedMemory
{
	// 重载类型转换，把int类型转换成T*类型
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// double偏特化，避免未对齐的内存导致的编译错误
template <>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double *address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	if (val == 0.0)
		return __longlong_as_double(old);
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

// 将g_idata归约到*sum
template <class T>
__global__ void Reduce(T* g_idata, size_t n, T* sum)
{
	cg::thread_block cta = cg::this_thread_block();
	T* sdata = SharedMemory<T>();
	uint32_t tid = threadIdx.x;
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t gridSize = blockDim.x * gridDim.x;
	uint32_t blockSize = blockDim.x / 2;

	T mySum = 0;
	// 每个线程reduce多个元素，取决于active blocks数 (gridDim)
	while (i < n) {
		mySum += g_idata[i];
		// 确保没有越界
		if (i + blockSize < n)
			mySum += g_idata[i + blockSize];
		i += gridSize;
	}
	// 每个线程将部分和放入shared memory
	sdata[tid] = mySum;
	cg::sync(cta);
	// shared mem中做归约
	if (tid < 256)
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	cg::sync(cta);
	if (tid < 128)
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	cg::sync(cta);
	if (tid < 64)
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	cg::sync(cta);
	if (tid < 32) {
		// 加上第2个warp
		mySum += sdata[tid + 32];
		cg::coalesced_group active = cg::coalesced_threads();
		// warp reduce
		for (uint32_t offset = 16; offset; offset >>= 1)
			mySum += active.shfl_down(mySum, offset);
	}
	// 每个block将自己的结果加入*sum
	if (tid == 0) {
		atomicAdd(sum, mySum);
		//printf("mySum = %f\n", mySum);
	}
}

#endif