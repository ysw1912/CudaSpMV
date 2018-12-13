#include "mmio.h"
#include "pagerank.h"

namespace test
{
	template <class ValueType>
	__global__ void warpSum(ValueType *input, int *seg, int n)
	{
		ValueType *space = SharedMemory<ValueType>();
		const uint32_t tid = threadIdx.x;
		const uint32_t laneId = threadIdx.x % 32; /* lane index in the warp */

		cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
		ValueType tmp = input[tid];
		if (tid < n) {
			ValueType saveTmp;
			if (seg[tid]) {
				saveTmp = tmp;
			}
			__syncwarp();
			for (int offset = 1; offset < 32; offset <<= 1) {
				ValueType t = tmp;
				t += tile32.shfl_up(t, offset);
				tile32.sync();
				if (laneId >= offset) {
					tmp = t;
				}
				tile32.sync();
			}
			space[tid] = tmp;
			__syncwarp();
			if (seg[tid]) {
				if (seg[tid] > 1) {
					ValueType segEnd = space[tid + seg[tid] - 1];
					tile32.sync();
					tmp = saveTmp + segEnd - tmp;
				}
				input[tid] = tmp;
			}
		}
	}

	void test01();
	void test02();
}

