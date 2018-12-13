/*
* LightSpMVCore.h
*
*  Created on: May 29, 2015
*      Author: Yongchao Liu
*      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
*      Email: yliu@cc.gatech.edu
*
*/
#ifndef __LIGHT_SPMV_CORE__
#define __LIGHT_SPMV_CORE__

#include "cpu_helper.h"
#include "gpu_helper.h"

namespace lightspmv
{

	/* 64位类型(如double)的shfl_down */
	template<typename T>
	__device__ inline T shfl_down_64bits(T var, int32_t srcLane, int32_t width)
	{
		int2 a = *reinterpret_cast<int2*>(&var);
		a.x = __shfl_down(a.x, srcLane, width);
		a.y = __shfl_down(a.y, srcLane, width);
		return *reinterpret_cast<T*>(&a);
	}

	/* 获取纹理内存/全局内存中的数组元素float */
	__device__ inline float FLOAT_VECTOR_GET(const cudaTextureObject_t vectorX, uint32_t index)
	{
		return tex1Dfetch<float>(vectorX, index);
	}
	__device__ inline float FLOAT_VECTOR_GET(const float* __restrict vectorX, uint32_t index)
	{
		return vectorX[index];
	}

	/* 获取纹理内存/全局内存中的数组元素double */
	__device__ inline double DOUBLE_VECTOR_GET(const cudaTextureObject_t vectorX, uint32_t index)
	{
		int2 v = tex1Dfetch<int2>(vectorX, index);
		return __hiloint2double(v.y, v.x);
	}
	__device__ inline double DOUBLE_VECTOR_GET(const double* __restrict vectorX, uint32_t index)
	{
		return vectorX[index];
	}

	/* 32位 y = αAx + βy */
	template<typename T, typename XType, uint32_t THREADS_PER_VECTOR, uint32_t VECTORS_PER_BLOCK>
	__global__ void csr32DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter,
										 const uint32_t cudaNumRows,
										 const uint32_t* __restrict rowOffsets,
										 const uint32_t* __restrict colIndexValues,
										 const T* __restrict numericalValues,
										 const XType vectorX,
										 T* vectorY,
										 const T alpha,
										 const T beta)
	{
		uint32_t i;
		T sum;
		uint32_t row;
		uint32_t rowStart, rowEnd;
		const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
		const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
		const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
		const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

		__shared__ volatile uint32_t space[VECTORS_PER_BLOCK][2];

		/* warp中的0号线程负责读取RowCounter中下N行的索引，N = 32 / THREADS_PER_VECTOR
		   保证即使各个warp处理的行数不同，每个warp也只执行1次atomicAdd() */
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/* 0号线程将行索引广播，每个线程计算其处理的行索引 */
		row = __shfl(row, 0) + warpVectorId;

		while (row < cudaNumRows) {
			/* vector中0、1号线程获取处理行的首尾 */
			if (laneId < 2) {
				space[vectorId][laneId] = rowOffsets[row + laneId];
			}
			rowStart = space[vectorId][0];
			rowEnd = space[vectorId][1];

			sum = 0;
			/* 点乘 */
			if (THREADS_PER_VECTOR == 32) { // 1个warp处理1行
				/*	确保对齐内存访问
					rowStart - (rowStart & (THREADS_PER_VECTOR - 1))求得的是
					不超过rowStart的最大的THREADS_PER_VECTOR的倍数
				*/
				i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;
				/* 非对齐部分 */
				if (i >= rowStart && i < rowEnd) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
				/* 对齐部分 */
				for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}
			else {
				/* vectorSize < 32，即1个warp处理多行，则不管有没有对齐，因为基本无法对齐 */
				for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}
			/* vector内规约 */
			//sum *= alpha;
			for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
				sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
			}

			/* vector中0号线程将规约sum给Y */
			if (laneId == 0) {
				//vectorY[row] = sum + beta * vectorY[row];
				vectorY[row] = sum;
			}

			/* 获取新的1行 */
			if (warpLaneId == 0) {
				row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
			}
			row = __shfl(row, 0) + warpVectorId;
		} /* while */
	}

	/* 64位 y = αAx + βy */
	template<typename T, typename XType, uint32_t THREADS_PER_VECTOR, uint32_t VECTORS_PER_BLOCK>
	__global__ void csr64DynamicWarpBLAS(uint32_t* __restrict cudaRowCounter,
		const uint32_t cudaNumRows,
		const uint32_t* __restrict rowOffsets,
		const uint32_t* __restrict colIndexValues,
		const T* __restrict numericalValues, const XType vectorX,
		T* vectorY, const T alpha, const T beta)
	{
		uint32_t i;
		T sum;
		uint32_t row;
		uint32_t rowStart, rowEnd;
		const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
		const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
		const uint32_t warpLaneId = threadIdx.x & 31; /*lane index in the warp*/
		const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR; /*vector index in the warp*/

		__shared__ volatile uint32_t space[VECTORS_PER_BLOCK][2];

		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		row = __shfl(row, 0) + warpVectorId;

		while (row < cudaNumRows) {
			if (laneId < 2) {
				space[vectorId][laneId] = rowOffsets[row + laneId];
			}
			rowStart = space[vectorId][0];
			rowEnd = space[vectorId][1];

			sum = 0;
			if (THREADS_PER_VECTOR == 32) {
				i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;
				if (i >= rowStart && i < rowEnd) {
					sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
				}
				for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}
			else {
				for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * DOUBLE_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}

			//sum *= alpha;
			for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
				sum += shfl_down_64bits<T>(sum, i, THREADS_PER_VECTOR);
			}

			if (laneId == 0) {
				//vectorY[row] = sum + beta * vectorY[row];
				vectorY[row] = sum;
			}
			if (warpLaneId == 0) {
				row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
			}
			row = __shfl(row, 0) + warpVectorId;
		} /* while */
	}

}/* namespace */

#endif