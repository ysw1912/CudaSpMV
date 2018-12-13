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

	/* 64λ����(��double)��shfl_down */
	template<typename T>
	__device__ inline T shfl_down_64bits(T var, int32_t srcLane, int32_t width)
	{
		int2 a = *reinterpret_cast<int2*>(&var);
		a.x = __shfl_down(a.x, srcLane, width);
		a.y = __shfl_down(a.y, srcLane, width);
		return *reinterpret_cast<T*>(&a);
	}

	/* ��ȡ�����ڴ�/ȫ���ڴ��е�����Ԫ��float */
	__device__ inline float FLOAT_VECTOR_GET(const cudaTextureObject_t vectorX, uint32_t index)
	{
		return tex1Dfetch<float>(vectorX, index);
	}
	__device__ inline float FLOAT_VECTOR_GET(const float* __restrict vectorX, uint32_t index)
	{
		return vectorX[index];
	}

	/* ��ȡ�����ڴ�/ȫ���ڴ��е�����Ԫ��double */
	__device__ inline double DOUBLE_VECTOR_GET(const cudaTextureObject_t vectorX, uint32_t index)
	{
		int2 v = tex1Dfetch<int2>(vectorX, index);
		return __hiloint2double(v.y, v.x);
	}
	__device__ inline double DOUBLE_VECTOR_GET(const double* __restrict vectorX, uint32_t index)
	{
		return vectorX[index];
	}

	/* 32λ y = ��Ax + ��y */
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

		/* warp�е�0���̸߳����ȡRowCounter����N�е�������N = 32 / THREADS_PER_VECTOR
		   ��֤��ʹ����warp�����������ͬ��ÿ��warpҲִֻ��1��atomicAdd() */
		if (warpLaneId == 0) {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		/* 0���߳̽��������㲥��ÿ���̼߳����䴦��������� */
		row = __shfl(row, 0) + warpVectorId;

		while (row < cudaNumRows) {
			/* vector��0��1���̻߳�ȡ�����е���β */
			if (laneId < 2) {
				space[vectorId][laneId] = rowOffsets[row + laneId];
			}
			rowStart = space[vectorId][0];
			rowEnd = space[vectorId][1];

			sum = 0;
			/* ��� */
			if (THREADS_PER_VECTOR == 32) { // 1��warp����1��
				/*	ȷ�������ڴ����
					rowStart - (rowStart & (THREADS_PER_VECTOR - 1))��õ���
					������rowStart������THREADS_PER_VECTOR�ı���
				*/
				i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;
				/* �Ƕ��벿�� */
				if (i >= rowStart && i < rowEnd) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
				/* ���벿�� */
				for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}
			else {
				/* vectorSize < 32����1��warp������У��򲻹���û�ж��룬��Ϊ�����޷����� */
				for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
					sum += numericalValues[i] * FLOAT_VECTOR_GET(vectorX, colIndexValues[i]);
				}
			}
			/* vector�ڹ�Լ */
			//sum *= alpha;
			for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
				sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
			}

			/* vector��0���߳̽���Լsum��Y */
			if (laneId == 0) {
				//vectorY[row] = sum + beta * vectorY[row];
				vectorY[row] = sum;
			}

			/* ��ȡ�µ�1�� */
			if (warpLaneId == 0) {
				row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
			}
			row = __shfl(row, 0) + warpVectorId;
		} /* while */
	}

	/* 64λ y = ��Ax + ��y */
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