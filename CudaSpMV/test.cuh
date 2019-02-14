#pragma once

#include "mmio.h"
#include "pagerank.h"

uint32_t next_pow_of_2(uint32_t x);
uint32_t iDivUp(uint32_t dividend, uint32_t divisor);

#define MAX_SCAN_ELEMENTS	67108864	// 2^6 * 2^20
#define BLOCK_SIZE			1024

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

	/* 神TM难！我太强啦~ */
	enum ScanType { Exclusive, Inclusive };
	void TestScan();
	void TestSegmentedScan();
}

