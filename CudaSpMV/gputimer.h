#include "cuda_runtime.h"

#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start, cudaEventBlockingSync);
		cudaEventCreate(&stop, cudaEventBlockingSync);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

#endif  /* __GPU_TIMER_H__ */
