#ifndef __BRC_SPMV_CORE__
#define __BRC_SPMV_CORE__

#include "cpu_helper.h"
#include "gpu_helper.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace brcspmv
{
	// BrcSpMV
	template <class ValueType>
	__global__ void brcSpMV(const uint32_t rep,
							const uint32_t B1,
							const uint32_t* rowPerm,
							const uint32_t* col,
							const ValueType* value,
							const uint32_t* blockPtr,
							const uint32_t* block_width,
							const uint32_t* numBlocks,
							const ValueType* x,
							ValueType* y)
	{
		uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t b_row = tid % B1;		// row of current block
		for (uint32_t i = 0; i < rep; ++i) {
			if (tid < *numBlocks * B1) {
				uint32_t cb = tid / B1;				// current block
				ValueType tmp = ValueType(0);
				for (uint32_t j = 0; j < block_width[cb]; ++j) {
					uint32_t index = blockPtr[cb] + j * B1 + b_row;
					tmp += value[index] * x[col[index]];
				}
				atomicAdd(y + rowPerm[tid], tmp);
				tid += gridDim.x * blockDim.x;
			}
		}
	}

	// BrcPlusSpMV
	template <typename ValueType>
	__global__ void brcPlusSpMV(const uint32_t rep,
								const uint32_t B1,
								const uint32_t numBlocks,
								const uint32_t *rowPerm,
								const uint32_t *rowSegLen,
								const uint32_t *col,
								const ValueType *value,
								const uint32_t *blockPtr,
								const ValueType *x,
								ValueType *y)
	{
		ValueType *space = SharedMemory<ValueType>();
		cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
		uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t tid = threadIdx.x;
		const uint32_t laneId = threadIdx.x % 32; /* lane index in the warp */
		const uint32_t b_row = id % B1;		// ��ǰ��ĵ�b_row��
		for (uint32_t i = 0; i < rep; ++i) {
			if (id < numBlocks * B1) {
				uint32_t cb = id / B1;	// ��cb����
				uint32_t blockWidth = (blockPtr[cb + 1] - blockPtr[cb]) / B1;	// ��cb����Ŀ��
				ValueType tmp = ValueType(0);
				for (uint32_t j = 0; j < blockWidth; ++j) {
					uint32_t index = blockPtr[cb] + j * B1 + b_row;
					tmp += value[index] * x[col[index]];
				}

				/* ��ÿ���̲߳��ֺ�tmp��segmented sum */
				ValueType saveTmp = tmp;	// ����tmpֵ
				for (uint32_t offset = 1; offset < 32; offset <<= 1) {
					ValueType t = tmp;
					t += tile32.shfl_up(t, offset);
					//tile32.sync();
					if (laneId >= offset) {
						tmp = t;
					}
					//tile32.sync();
				}
				space[tid] = tmp;	// ��ÿ��warp��scan���빲���ڴ汸��
				//tile32.sync();
				if (rowSegLen[id]) {
					ValueType segEnd = space[tid + rowSegLen[id] - 1];	// ��βֵ
					tmp = saveTmp + segEnd - tmp;
					atomicAdd(y + rowPerm[id], tmp);	// ����ͬ�еĲ��ֺ�ԭ�Ӽӵ�y����
				}

				id += gridDim.x * blockDim.x;
			}
		} /* for */
	}

}/* namespace brcspmv */

#endif