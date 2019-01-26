#include "test.h"

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>

#if defined(__cplusplus) && defined(__CUDACC__)
# include "cooperative_groups_helpers.h"
namespace cg = cooperative_groups;
#endif

namespace test
{
	// x是否是2的幂次
	static inline bool is_pow_of_2(uint32_t x)
	{
		return !(x & (x - 1));
	}

	// 大于等于x的最小2的幂次数
	static inline uint32_t next_pow_of_2(uint32_t x)
	{
		if (is_pow_of_2(x))
			return x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return x + 1;
	}

	// BRC矩阵的一行
	struct st
	{
		uint32_t a;
		uint32_t b;
		uint32_t c;

		__host__ __device__ bool operator<(const st &rhs) const
		{
			return this->b > rhs.b;
		}
	};

	void test01()
	{
		int N = 100000;
		thrust::host_vector<st> v(N);
		for (int i = 0; i < N; ++i) {
			v[i].a = 0;
			v[i].b = N - i;
			v[i].c = i;
		}
		printf("1\n");
		thrust::device_vector<st> dv(N);
		printf("2\n");
		thrust::copy(v.begin(), v.end(), dv.begin());
		printf("3\n");
		thrust::stable_sort(dv.begin(), dv.end());
		printf("4\n");
		v = dv;
		printf("5\n");
		for (int i = 0; i < 7; i++) {
			printf("(%d - %d - %d), ", v[i].a, v[i].b, v[i].c);
		}
		printf("\n");
	}

	extern "C" const uint32_t MAX_BATCH_ELEMENTS = 64 * 1048576;	// 2^6 * 2^20
#define BLOCK_SIZE 1024

	void test02()
	{
		uint32_t *d_Input, *d_Output;
		uint32_t *h_Input, *h_OutputCPU, *h_OutputGPU;
		uint32_t N;
		scanf("%d", &N);
		uint32_t d_N = next_pow_of_2(N);
		if (d_N < 4 * BLOCK_SIZE)
			d_N = 4 * BLOCK_SIZE;

		h_Input = (uint32_t*)malloc(d_N * sizeof(uint32_t));
		h_OutputCPU = (uint32_t*)malloc(N * sizeof(uint32_t));
		h_OutputGPU = (uint32_t*)malloc(N * sizeof(uint32_t));
		for (uint32_t i = 0; i < N; i++)
			h_Input[i] = 1;
		for (uint32_t i = N; i < d_N; i++)
			h_Input[i] = 0;

		checkCudaError(cudaMalloc((void**)&d_Input, d_N * sizeof(uint32_t)));
		checkCudaError(cudaMalloc((void**)&d_Output, d_N * sizeof(uint32_t)));
		checkCudaError(cudaMemcpy(d_Input, h_Input, d_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
		ExclusiveScanHost(h_OutputCPU, h_Input, N);
		printf("[CPU] "); Print<uint32_t>(h_OutputCPU, N, false);

		printf("Running scan...\n[CPU arrayLength: %u, GPU arrayLength: %u]\n", N, d_N);
		checkCudaError(cudaDeviceSynchronize());
		ExclusiveScan(d_Output, d_Input, d_N);
		checkCudaError(cudaDeviceSynchronize());

		// 只传出N个元素
		checkCudaError(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		printf("[GPU] "); Print<uint32_t>(h_OutputGPU, N, false);

		checkCudaError(cudaFree(d_Input));
		checkCudaError(cudaFree(d_Output));

		printf("Validating the results...\n");
		int flag = 1;
		for (uint32_t i = 0; i < N; i++) {
			if (h_OutputCPU[i] != h_OutputGPU[i]) {
				flag = 0;
				break;
			}
		}
		printf(" ...Results %s\n\n", (flag == 1) ? "Match" : "DON'T Match !!!");
	}

	/*
	** return 因子factor
	** factor * 2^log2L = L
	** 如L=5, 则log2L=0, factor=5, 因为5 * 2^0 = 5
	** 如L=6, 则log2L=1, factor=3, 因为3 * 2^1 = 6
	**/
	static uint32_t factorRadix2(uint32_t &log2L, uint32_t L)
	{
		if (!L) {
			log2L = 0;
			return 0;
		}
		else {
			for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
			return L;
		}
	}

	/*
	** return 除法结果
	** 除不尽则向上取整
	** 如 5 / 2 返回 3
	**/
	static uint32_t iDivUp(uint32_t dividend, uint32_t divisor)
	{
		return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
	}

	void ExclusiveScanHost(uint32_t* dst, uint32_t* src, uint32_t size)
	{
		dst[0] = 0;
		for (uint32_t i = 1; i < size; i++)
			dst[i] = src[i - 1] + dst[i - 1];			
	}

	inline __device__
	uint32_t scan1Inclusive(uint32_t idata, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
	{
		uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
		s_Data[pos] = 0;
		pos += size;
		s_Data[pos] = idata;
#pragma unroll
		for (uint32_t offset = 1; offset < size; offset <<= 1) {
			cg::sync(cta);
			uint32_t t = s_Data[pos] + s_Data[pos - offset];
			cg::sync(cta);
			s_Data[pos] = t;
		}

		return s_Data[pos];
	}

	inline __device__
	uint32_t scan1Exclusive(uint32_t idata, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
	{
		return scan1Inclusive(idata, s_Data, size, cta) - idata;
	}


	inline __device__
	uint4 scan4Inclusive(uint4 idata4, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
	{
		//Level-0 inclusive scan
		idata4.y += idata4.x;
		idata4.z += idata4.y;
		idata4.w += idata4.z;

		//Level-1 exclusive scan
		uint32_t oval = scan1Exclusive(idata4.w, s_Data, size, cta);
		//printf("%d %d\n", idata4.w, oval);

		idata4.x += oval;
		idata4.y += oval;
		idata4.z += oval;
		idata4.w += oval;
		
		return idata4;
	}

	inline __device__
	uint4 scan4Exclusive(uint4 idata4, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
	{
		uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
		odata4.x -= idata4.x;
		odata4.y -= idata4.y;
		odata4.z -= idata4.z;
		odata4.w -= idata4.w;
		return odata4;
	}

	__global__ void ExclusiveScanShared(uint4 *d_Dst, uint4 *d_Src, uint32_t size)
	{
		cg::thread_block cta = cg::this_thread_block();
		//__shared__ uint32_t s_Data[2 * BLOCK_SIZE];
		uint32_t* s_Data = SharedMemory<uint32_t>();

		uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
		uint4 idata4 = d_Src[pos];
		uint4 odata4 = scan4Exclusive(idata4, s_Data, size, cta);
		d_Dst[pos] = odata4;
	}

	// 对 list[每(4 * BLOCK_SIZE)个元素的和] 做exclusive scan
	__global__ void ExclusiveScanShared2(uint32_t* d_Buf, uint32_t* d_Dst, uint32_t* d_Src, uint32_t N, uint32_t size)
	{
		cg::thread_block cta = cg::this_thread_block();
		__shared__ uint32_t s_Data[2 * BLOCK_SIZE];

		uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

		uint32_t idata = 0;
		if (pos < N)	// 跳过读写最后一个block的非活跃线程
			// 从d_Dst读取每(4 * BLOCK_SIZE)中最后一个元素，即为exclusive scan的最大结果
			// 加上d_Src相应元素得到inclusive scan的结果，即该(4 * BLOCK_SIZE)个元素的和
			idata = d_Dst[(4 * BLOCK_SIZE) - 1 + (4 * BLOCK_SIZE) * pos]
				  + d_Src[(4 * BLOCK_SIZE) - 1 + (4 * BLOCK_SIZE) * pos];

		// 对idata做exclusive scan
		uint32_t odata = scan1Exclusive(idata, s_Data, size, cta);

		if (pos < N) {
			d_Buf[pos] = odata;
			//printf("idata = %d, odata = %d\n", idata, odata);
		}
	}

	// 将d_Buf中每个block的结果加至d_Dst
	__global__ void UniformUpdate(uint4 *d_Dst, uint32_t *d_Buf)
	{
		cg::thread_block cta = cg::this_thread_block();
		__shared__ uint32_t buf;
		uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

		if (threadIdx.x == 0)
			buf = d_Buf[blockIdx.x];
		cg::sync(cta);

		uint4 data4 = d_Dst[pos];
		data4.x += buf;
		data4.y += buf;
		data4.z += buf;
		data4.w += buf;
		d_Dst[pos] = data4;
	}

	void ExclusiveScanShort(uint32_t *d_Dst, uint32_t *d_Src, uint32_t size)
	{
		// 检查是否所有block包含数据
		assert(size % (4 * BLOCK_SIZE) == 0);

		ExclusiveScanShared<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(uint32_t) >>>((uint4*)d_Dst, (uint4*)d_Src, size / 4);
	}

	void ExclusiveScanLarge(uint32_t *d_Dst, uint32_t *d_Src, uint32_t size)
	{
		// 一个block处理(4 * BLOCK_SIZE)个元素的exclusive scan
		ExclusiveScanShared<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(uint32_t)>>>((uint4*)d_Dst, (uint4*)d_Src, BLOCK_SIZE);

		uint32_t *d_Buf;	// 存放ExclusiveScanShared2的结果
		checkCudaError(cudaMalloc((void**)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * BLOCK_SIZE)) * sizeof(uint32_t)));

		const uint32_t blockCount2 = iDivUp(size / (4 * BLOCK_SIZE), BLOCK_SIZE);
		ExclusiveScanShared2<<<blockCount2, BLOCK_SIZE>>>(d_Buf, d_Dst, d_Src, size / (4 * BLOCK_SIZE), size / (4 * BLOCK_SIZE));

		UniformUpdate<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Dst, d_Buf);
		
		checkCudaError(cudaFree(d_Buf));
	}

	void ExclusiveScan(uint32_t* d_Dst, uint32_t* d_Src, uint32_t size)
	{
		// 检查size是否是2的幂次
		assert(is_pow_of_2(size));
		//uint32_t log2L;
		//assert(factorRadix2(log2L, size) == 1);

		if (size <= 4 * BLOCK_SIZE)
			ExclusiveScanShort(d_Dst, d_Src, size);
		else
			ExclusiveScanLarge(d_Dst, d_Src, size);
	}
}