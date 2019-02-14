#include "test.cuh"

/*
** return 因子factor
** factor * 2^log2L = L
** 如L=5, 则log2L=0, factor=5, 因为5 * 2^0 = 5
** 如L=6, 则log2L=1, factor=3, 因为3 * 2^1 = 6
**/
/*
uint32_t factorRadix2(uint32_t &log2L, uint32_t L)
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
*/

/*
** return 除法结果
** 除不尽则向上取整
** 如 5 / 2 返回 3
**/
uint32_t iDivUp(uint32_t dividend, uint32_t divisor)
{
	return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

// x是否是2的幂次
bool is_pow_of_2(uint32_t x)
{
	return !(x & (x - 1));
}

// 大于等于x的最小2的幂次数
uint32_t next_pow_of_2(uint32_t x)
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

namespace test
{
	/************************ CPU Scan ***********************/
	void ScanHost(uint32_t* dst, uint32_t* src, uint32_t size, ScanType st)
	{
		if (st == Exclusive) {
			dst[0] = 0;
			for (uint32_t i = 1; i < size; i++)
				dst[i] = src[i - 1] + dst[i - 1];
		}
		else {
			dst[0] = src[0];
			for (uint32_t i = 1; i < size; i++)
				dst[i] = src[i] + dst[i - 1];
		}
	}

	void SegScanHost(uint32_t* dst, uint32_t* src, uint32_t* flag, uint32_t size, ScanType st)
	{
		if (st == Exclusive) {
			for (uint32_t i = 0; i < size; ++i) {
				if (flag[i] == 1)
					dst[i] = 0;
				else
					dst[i] = src[i - 1] + dst[i - 1];
			}
		}
		else {
			for (uint32_t i = 0; i < size; ++i) {
				if (flag[i] == 1)
					dst[i] = src[i];
				else
					dst[i] = src[i] + dst[i - 1];
			}
		}
	}

	/******************** Scan ********************/
	
	__device__ uint32_t scan1Inclusive(uint32_t idata, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
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
		// pos = 1024~2047, s_Data[pos] = 4 8 12 16...
		// printf("s_Data[%d] = %d\n", pos, s_Data[pos]);
		return s_Data[pos];
	}

	__device__ uint32_t scan1Exclusive(uint32_t idata, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
	{
		return scan1Inclusive(idata, s_Data, size, cta) - idata;
	}

	__device__ uint4 scan4Inclusive(uint4 idata4, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
	{
		//Level-0 inclusive scan
		idata4.y += idata4.x;
		idata4.z += idata4.y;
		idata4.w += idata4.z;

		//Level-1 exclusive scan
		uint32_t oval = scan1Exclusive(idata4.w, s_Data, size, cta);
		// idata4.y = 2, idata4.z = 3, idata4.w = 4
		// printf("idata4.w = %d, oval = %d\n", idata4.w, oval);

		idata4.x += oval;
		idata4.y += oval;
		idata4.z += oval;
		idata4.w += oval;

		return idata4;
	}

	__device__ uint4 scan4Exclusive(uint4 idata4, volatile uint32_t *s_Data, uint32_t size, cg::thread_block cta)
	{
		uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
		odata4.x -= idata4.x;
		odata4.y -= idata4.y;
		odata4.z -= idata4.z;
		odata4.w -= idata4.w;
		return odata4;
	}

	__global__ void ScanShared(uint4 *d_Dst, uint4 *d_Src, uint32_t size, ScanType st)
	{
		cg::thread_block cta = cg::this_thread_block();
		__shared__ uint32_t s_Data[2 * BLOCK_SIZE];

		uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
		uint4 idata4 = d_Src[pos];
		uint4 odata4;
		if (st == Exclusive)
			odata4 = scan4Exclusive(idata4, s_Data, size, cta);
		else
			odata4 = scan4Inclusive(idata4, s_Data, size, cta);
		d_Dst[pos] = odata4;
	}

	// 对 list[每(4 * BLOCK_SIZE)个元素的和] 做scan
	__global__ void ScanShared2(uint32_t* d_Buf, uint32_t* d_Dst, uint32_t* d_Src, uint32_t N, ScanType st)
	{
		cg::thread_block cta = cg::this_thread_block();
		__shared__ uint32_t s_Data[2 * BLOCK_SIZE];

		uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

		uint32_t idata = 0;
		if (pos < N) {	// 跳过读取最后一个block的结果
						// 从d_Dst读取每(4 * BLOCK_SIZE)中最后一个元素，即为ScanShared的最大结果
						// 加上d_Src相应元素得到inclusive scan的结果，即该(4 * BLOCK_SIZE)个元素的和
			idata = d_Dst[(4 * BLOCK_SIZE) - 1 + (4 * BLOCK_SIZE) * pos]
				+ (st == Exclusive) * d_Src[(4 * BLOCK_SIZE) - 1 + (4 * BLOCK_SIZE) * pos];
		}

		// 对idata做scan
		uint32_t odata;
		odata = scan1Exclusive(idata, s_Data, N, cta);

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

	void ScanShort(uint32_t *d_Dst, uint32_t *d_Src, uint32_t size, ScanType st)
	{
		// 检查是否所有block包含数据
		assert(size % (4 * BLOCK_SIZE) == 0);

		ScanShared<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Dst, (uint4*)d_Src, size / 4, st);
	}

	void ScanLarge(uint32_t *d_Dst, uint32_t *d_Src, uint32_t size, ScanType st)
	{
		// 一个block处理(4 * BLOCK_SIZE)个元素的scan
		ScanShared<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Dst, (uint4*)d_Src, BLOCK_SIZE, st);

		uint32_t *d_Buf;	// 存放ScanShared2的结果
		checkCudaError(cudaMalloc((void**)&d_Buf, (MAX_SCAN_ELEMENTS / (4 * BLOCK_SIZE)) * sizeof(uint32_t)));

		const uint32_t blockCount2 = iDivUp(size / (4 * BLOCK_SIZE), BLOCK_SIZE);
		ScanShared2<<<blockCount2, BLOCK_SIZE>>>(d_Buf, d_Dst, d_Src, size / (4 * BLOCK_SIZE), st);

		UniformUpdate<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Dst, d_Buf);

		checkCudaError(cudaFree(d_Buf));
	}

	void Scan(uint32_t* d_Dst, uint32_t* d_Src, uint32_t size, ScanType st)
	{
		// 检查size是否是2的幂次
		assert(is_pow_of_2(size));
		//uint32_t log2L;
		//assert(factorRadix2(log2L, size) == 1);

		if (size <= 4 * BLOCK_SIZE)
			ScanShort(d_Dst, d_Src, size, st);
		else
			ScanLarge(d_Dst, d_Src, size, st);
	}

	void TestScan()
	{
		uint32_t *d_Input, *d_Output;
		uint32_t *h_Input, *h_OutputCPU, *h_OutputGPU;
		uint32_t N;
		scanf("%d", &N);
		getchar();
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

		h_Input[45] = 3;		h_Input[37] = 4;		h_Input[91] = 7;
		h_Input[1045] = 2;		h_Input[1037] = 5;		h_Input[1091] = 5;
		h_Input[2045] = 6;		h_Input[2037] = 4;		h_Input[2091] = 3;

		char c;
		printf("Input ScanType: ");
		scanf("%c", &c);
		if (c == 'e')
			ScanHost(h_OutputCPU, h_Input, N, Exclusive);
		else if (c == 'i')
			ScanHost(h_OutputCPU, h_Input, N, Inclusive);
		printf("[CPU] ");
		Print<uint32_t>(h_OutputCPU, N, false);

		checkCudaError(cudaMalloc((void**)&d_Input, d_N * sizeof(uint32_t)));
		checkCudaError(cudaMemcpy(d_Input, h_Input, d_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
		checkCudaError(cudaMalloc((void**)&d_Output, d_N * sizeof(uint32_t)));

		free(h_Input);

		printf("Running scan...\n[CPU arrayLength: %u, GPU arrayLength: %u]\n", N, d_N);

		if (c == 'e')
			Scan(d_Output, d_Input, d_N, Exclusive);
		else if (c == 'i')
			Scan(d_Output, d_Input, d_N, Inclusive);

		// 只传出N个元素
		checkCudaError(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		printf("[GPU] ");
		Print<uint32_t>(h_OutputGPU, N, false);

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
		free(h_OutputCPU);
		free(h_OutputGPU);
		printf(" ...Results %s\n\n", (flag == 1) ? "Match" : "DON'T Match !!!");
	}

	/******************** Segmented Scan ********************/

	inline __device__
	uint32_t segScan1Inclusive(uint32_t idata, uint32_t iflag,
							   volatile uint32_t *s_Data, volatile uint32_t *s_Flag,
							   uint32_t size, cg::thread_block cta)
	{
		const uint32_t id = threadIdx.x & (size - 1);
		uint32_t pos = 2 * threadIdx.x - id;
		if (pos < size) {
			s_Data[pos] = 0;
			s_Flag[pos] = 0;
			pos += size;
			s_Data[pos] = idata;
			s_Flag[pos] = iflag * id;
#pragma unroll
			for (uint32_t offset = 1; offset < size; offset <<= 1) {
				cg::sync(cta);
				uint32_t f = (s_Flag[pos] > s_Flag[pos - offset]) * s_Flag[pos]
					+ (s_Flag[pos] <= s_Flag[pos - offset]) * s_Flag[pos - offset];
				cg::sync(cta);
				s_Flag[pos] = f;	// min_index
			}
			//printf("s_Flag[%d] = %d, s_Flag[%d] = %d\n", pos - size, s_Flag[pos - size], pos, s_Flag[pos]);
			//printf("s_Data[%d] = %d, s_Data[%d] = %d\n", pos - size, s_Data[pos - size], pos, s_Data[pos]);
			// pos = 1024~2047
			// s_Flag[pos] = 0 0 ... 256 256 ... 512 512 ... 768 768 ...
			//printf("s_Flag[%d] = %d\n", pos, s_Flag[pos]);
#pragma unroll
			for (uint32_t offset = 1; offset < size; offset <<= 1) {
				cg::sync(cta);
				uint32_t v = (id >= (s_Flag[pos] + offset)) * (s_Data[pos] + s_Data[pos - offset])
					+ (id < (s_Flag[pos] + offset)) * s_Data[pos];
				cg::sync(cta);
				s_Data[pos] = v;
			}
			//printf("s_Flag[%d] = %d, s_Flag[%d] = %d\n", pos - size, s_Flag[pos - size], pos, s_Flag[pos]);
			//printf("s_Data[%d] = %d, s_Data[%d] = %d\n", pos - size, s_Data[pos - size], pos, s_Data[pos]);
		}
		return s_Data[pos];
	}

	inline __device__
	uint32_t segScan1Exclusive(uint32_t idata, uint32_t iflag,
							   volatile uint32_t *s_Data, volatile uint32_t *s_Flag,
							   uint32_t size, cg::thread_block cta)
	{
		return segScan1Inclusive(idata, iflag, s_Data, s_Flag, size, cta) - idata;
	}


	inline __device__
	uint4 segScan4Inclusive(uint4 idata4, uint4 iflag4, 
		                    volatile uint32_t *s_Data, volatile uint32_t *s_Flag,
						    uint32_t size, cg::thread_block cta)
	{
		bool isOpen = (iflag4.x == 0);

		// Step 1: 将headFlags转换为minimum_index的形式
		iflag4.z *= 2;
		iflag4.w *= 3;

		// 对iflag4做Inclusive Scan, 得到的midx4用于确定iflag以及Step 4是否加上一个线程的结果
		uint4 midx4;
		midx4.x = 0;
		midx4.y = iflag4.y;
		midx4.z = (midx4.y > iflag4.z) * midx4.y + (midx4.y <= iflag4.z) * iflag4.z;
		midx4.w = (midx4.z > iflag4.w) * midx4.z + (midx4.z <= iflag4.w) * iflag4.w;

		// Step 2: 对4个元素做segmented scan
		idata4.y += (iflag4.y == 0) * idata4.x;
		idata4.z += (iflag4.z == 0) * idata4.y;
		idata4.w += (iflag4.w == 0) * idata4.z;

		//printf("[%d]: %d %d %d %d\n", threadIdx.x, idata4.x, idata4.y, idata4.z, idata4.w);

		// Step 3: 对每4个元素的结果(共1024个结果)做Inclusive Segmented Scan
		uint32_t iflag = (midx4.w != 0) || !isOpen;
		segScan1Inclusive(idata4.w, iflag, s_Data, s_Flag, size, cta);
		// 获取前一个线程的scan结果
		uint32_t oval = s_Data[threadIdx.x + size - 1];

		// Step 4: 将Step 3的结果加进来
		if (isOpen) {
			idata4.x += oval;
			idata4.y += (midx4.y == 0) * oval;
			idata4.z += (midx4.z == 0) * oval;
			idata4.w += (midx4.w == 0) * oval;
		}
		return idata4;
	}

	inline __device__
	uint4 segScan4Exclusive(uint4 idata4, uint4 iflag4,
							volatile uint32_t *s_Data, volatile uint32_t *s_Flag,
							uint32_t size, cg::thread_block cta)
	{
		uint4 odata4 = segScan4Inclusive(idata4, iflag4, s_Data, s_Flag, size, cta);
		odata4.x -= idata4.x;
		odata4.y -= idata4.y;
		odata4.z -= idata4.z;
		odata4.w -= idata4.w;
		return odata4;
	}

	__global__ void SegScanShared(uint4 *d_Dst, uint4 *d_Src, uint4 *d_Flag, uint32_t size, ScanType st)
	{
		cg::thread_block cta = cg::this_thread_block();
		__shared__ uint32_t s_Data[2 * BLOCK_SIZE];
		__shared__ uint32_t s_Flag[2 * BLOCK_SIZE];
		//uint32_t* s_Data = SharedMemory<uint32_t>();

		const uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
		uint4 idata4 = d_Src[pos], iflag4 = d_Flag[pos];
		uint4 odata4;
		if (st == Exclusive)
			odata4 = segScan4Exclusive(idata4, iflag4, s_Data, s_Flag, size, cta);
		else
			odata4 = segScan4Inclusive(idata4, iflag4, s_Data, s_Flag, size, cta);
		d_Dst[pos] = odata4;
	}

	void SegScanShort(uint32_t *d_Dst, uint32_t *d_Src, uint32_t* d_HeadFlag, uint32_t size, ScanType st)
	{
		// 检查是否所有block包含数据
		assert(size % (4 * BLOCK_SIZE) == 0);

		SegScanShared<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Dst, (uint4*)d_Src, (uint4*)d_HeadFlag, size / 4, st);
	}

	__global__ void MidxInit(uint4* d_Midx, uint4* d_HeadFlag, uint32_t size)
	{
		const uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
		const uint32_t stride = pos * 4;
		uint4 flag = d_HeadFlag[pos];
		uint4 midx;
		midx.x = (flag.x == 1) * stride;
		midx.y = (flag.y == 1) * (1 + stride);
		midx.z = (flag.z == 1) * (2 + stride);
		midx.w = (flag.w == 1) * (3 + stride);
		d_Midx[pos] = midx;
	}

	__global__ void SegScanL(uint32_t* d_BlockFlag, uint4* d_Dst, uint4* d_Src, uint4* d_Flag, uint32_t size, ScanType st)
	{
		cg::thread_block cta = cg::this_thread_block();
		__shared__ uint32_t s_Data[2 * BLOCK_SIZE];
		__shared__ uint32_t s_Flag[2 * BLOCK_SIZE];
		//uint32_t* s_Data = SharedMemory<uint32_t>();

		const uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
		uint4 idata4 = d_Src[pos], iflag4 = d_Flag[pos];
		uint4 odata4;
		cg::sync(cta);
		if (st == Exclusive)
			odata4 = segScan4Exclusive(idata4, iflag4, s_Data, s_Flag, size, cta);
		else
			odata4 = segScan4Inclusive(idata4, iflag4, s_Data, s_Flag, size, cta);
		//
		cg::sync(cta);
		if (threadIdx.x == 0) {
			uint32_t flag = (s_Flag[2 * BLOCK_SIZE - 1] != 0) || d_Flag[pos].x == 1;
			d_BlockFlag[blockIdx.x] = flag;
			//printf("[Block %d]: s_Flag[last] = %d, d_BlockFlag = %d\n", blockIdx.x, s_Flag[2 * BLOCK_SIZE - 1], flag);
		}

		d_Dst[pos] = odata4;
	}


	// 对 [每个block的结果] 做scan
	__global__ void SegScanShared2(uint32_t* d_Dst, uint32_t* d_Src, uint32_t* d_Data, uint32_t* d_BlockFlag, uint32_t N, ScanType st)
	{
		cg::thread_block cta = cg::this_thread_block();
		__shared__ uint32_t s_Data[2 * BLOCK_SIZE];
		__shared__ uint32_t s_Flag[2 * BLOCK_SIZE];

		uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
		uint32_t idata = 0, iflag = 0;
		if (pos < N) {	// 跳过读取最后一个block(N = size / (4 * BLOCK_SIZE))
			//printf("Block[%d] = %d\n", pos, d_Dst[(4 * BLOCK_SIZE) - 1 + (4 * BLOCK_SIZE) * pos]);
			// 从d_Dst读取每(4 * BLOCK_SIZE)中最后一个元素，即为ScanShared的最大结果
			// 加上d_Src相应元素得到inclusive scan的结果，即该(4 * BLOCK_SIZE)个元素的和
			idata = d_Dst[(4 * BLOCK_SIZE) - 1 + (4 * BLOCK_SIZE) * pos]
				  + (st == Exclusive) * d_Src[(4 * BLOCK_SIZE) - 1 + (4 * BLOCK_SIZE) * pos];
			iflag = d_BlockFlag[pos];
		}

		// 对idata做scan
		segScan1Inclusive(idata, iflag, s_Data, s_Flag, N, cta);
		// 获取前一个线程的scan结果
		uint32_t odata = s_Data[threadIdx.x + N - 1];

		if (pos < N) {
			d_Data[pos] = odata;
			//printf("idata = %d, iflag = %d, odata = %d\n", idata, iflag, odata);
		}
	}

	// 将d_Data中每个block的结果加至d_Dst
	__global__ void SegmentUpdate(uint4* d_Dst, uint4* d_Midx, uint32_t* d_Data)
	{
		cg::thread_block cta = cg::this_thread_block();
		uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

		__shared__ uint32_t buf;
		if (threadIdx.x == 0)
			buf = d_Data[blockIdx.x];
		cg::sync(cta);

		uint4 data4 = d_Dst[pos], midx4 = d_Midx[pos];
		data4.x += (midx4.x == 0) * buf;
		data4.y += (midx4.y == 0) * buf;
		data4.z += (midx4.z == 0) * buf;
		data4.w += (midx4.w == 0) * buf;
		d_Dst[pos] = data4;
	}

	void SegScanLarge(uint32_t *d_Dst, uint32_t *d_Src, uint32_t* d_HeadFlag, uint32_t size, ScanType st)
	{
		uint32_t *d_Midx;			// 存放d_HeadFlag的MAX Scan结果
		checkCudaError(cudaMalloc((void**)&d_Midx, size * sizeof(uint32_t)));
		MidxInit<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Midx, (uint4*)d_HeadFlag, size / 4);
		SegScanShared<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Midx, (uint4*)d_Midx, (uint4*)d_HeadFlag, BLOCK_SIZE, Inclusive);

		uint32_t* d_BlockFlag;	// 存放每个block的flag
		checkCudaError(cudaMalloc((void**)&d_BlockFlag, (MAX_SCAN_ELEMENTS / (4 * BLOCK_SIZE)) * sizeof(uint32_t)));
		// 一个block处理(4 * BLOCK_SIZE)个元素的scan
		SegScanL<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>(d_BlockFlag, (uint4*)d_Dst, (uint4*)d_Src, (uint4*)d_HeadFlag, BLOCK_SIZE, st);

		uint32_t* d_Data;	// 存放SegScanShared2的结果
		checkCudaError(cudaMalloc((void**)&d_Data, (MAX_SCAN_ELEMENTS / (4 * BLOCK_SIZE)) * sizeof(uint32_t)));

		const uint32_t gridDim = iDivUp(size / (4 * BLOCK_SIZE), BLOCK_SIZE);
		SegScanShared2<<<gridDim, BLOCK_SIZE>>>(d_Dst, d_Src, d_Data, d_BlockFlag, size / (4 * BLOCK_SIZE), st);

		checkCudaError(cudaFree(d_BlockFlag));

		SegmentUpdate<<<size / (4 * BLOCK_SIZE), BLOCK_SIZE>>>((uint4*)d_Dst, (uint4*)d_Midx, d_Data);

		checkCudaError(cudaFree(d_Data));
		checkCudaError(cudaFree(d_Midx));
	}

	void SegScan(uint32_t* d_Dst, uint32_t* d_Src, uint32_t* d_HeadFlag, uint32_t size, ScanType st)
	{
		// 检查size是否是2的幂次
		assert(is_pow_of_2(size));

		if (size <= 4 * BLOCK_SIZE)
			SegScanShort(d_Dst, d_Src, d_HeadFlag, size, st);
		else
			SegScanLarge(d_Dst, d_Src, d_HeadFlag, size, st);
	}

	void TestSegmentedScan()
	{
		uint32_t *d_Input, *d_Output, *d_HeadFlag;
		uint32_t *h_Input, *h_OutputCPU, *h_OutputGPU, *h_HeadFlag;
		uint32_t N;
		scanf("%d", &N);
		getchar();
		uint32_t d_N = next_pow_of_2(N);
		if (d_N < 4 * BLOCK_SIZE)
			d_N = 4 * BLOCK_SIZE;

		h_Input = (uint32_t*)malloc(d_N * sizeof(uint32_t));
		h_HeadFlag = (uint32_t*)malloc(d_N * sizeof(uint32_t));
		h_OutputCPU = (uint32_t*)malloc(N * sizeof(uint32_t));
		h_OutputGPU = (uint32_t*)malloc(N * sizeof(uint32_t));
		for (uint32_t i = 0; i < N; i++) {
			h_Input[i] = 1;
			if (i < 4096 && i % 1024 == 0)
				h_HeadFlag[i] = 1;
			else
				h_HeadFlag[i] = 0;
		}
		for (uint32_t i = N; i < d_N; i++) {
			h_Input[i] = 0;
			h_HeadFlag[i] = 0;
		}
		h_HeadFlag[N] = 1;

		h_HeadFlag[11] = 1;		h_HeadFlag[103] = 1;	h_HeadFlag[120] = 1;
		h_HeadFlag[315] = 1;	h_HeadFlag[613] = 1;	h_HeadFlag[919] = 1;
		h_HeadFlag[1090] = 1;	h_HeadFlag[1165] = 1;	h_HeadFlag[1874] = 1;
		h_HeadFlag[2318] = 1;	h_HeadFlag[2901] = 1;	h_HeadFlag[2999] = 1;
		h_HeadFlag[3096] = 1;	h_HeadFlag[3377] = 1;	h_HeadFlag[3674] = 1;
		h_HeadFlag[3712] = 1;	h_HeadFlag[4094] = 1;	h_HeadFlag[4095] = 1;

		h_HeadFlag[4100] = 1;	h_HeadFlag[4236] = 1;	h_HeadFlag[5173] = 1;
		h_HeadFlag[6245] = 1;	h_HeadFlag[6789] = 1;	h_HeadFlag[7534] = 1;
		h_HeadFlag[7933] = 1;	h_HeadFlag[8189] = 1;	h_HeadFlag[8190] = 1;

		h_HeadFlag[12289] = 1;	h_HeadFlag[12290] = 1;	h_HeadFlag[12901] = 1;
		h_HeadFlag[13579] = 1;	h_HeadFlag[14066] = 1;	h_HeadFlag[14999] = 1;

		char c;
		printf("Input ScanType: ");
		scanf("%c", &c);
		if (c == 'e')
			SegScanHost(h_OutputCPU, h_Input, h_HeadFlag, N, Exclusive);
		else if (c == 'i')
			SegScanHost(h_OutputCPU, h_Input, h_HeadFlag, N, Inclusive);
		printf("[CPU] ");
		PrintByRow<uint32_t>(h_OutputCPU, N, 1024);
		//Print<uint32_t>(h_OutputCPU, N);

		checkCudaError(cudaMalloc((void**)&d_Input, d_N * sizeof(uint32_t)));
		checkCudaError(cudaMemcpy(d_Input, h_Input, d_N * sizeof(uint32_t), cudaMemcpyHostToDevice));
		checkCudaError(cudaMalloc((void**)&d_Output, d_N * sizeof(uint32_t)));
		checkCudaError(cudaMalloc((void**)&d_HeadFlag, d_N * sizeof(uint32_t)));
		checkCudaError(cudaMemcpy(d_HeadFlag, h_HeadFlag, d_N * sizeof(uint32_t), cudaMemcpyHostToDevice));

		free(h_Input);
		//free(h_HeadFlag);	// 蜜汁报错？？？

		printf("Running scan...\n[CPU arrayLength: %u, GPU arrayLength: %u]\n", N, d_N);

		if (c == 'e')
			SegScan(d_Output, d_Input, d_HeadFlag, d_N, Exclusive);
		else if (c == 'i')
			SegScan(d_Output, d_Input, d_HeadFlag, d_N, Inclusive);

		// 只传出N个元素
		checkCudaError(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		printf("[GPU] ");
		PrintByRow<uint32_t>(h_OutputGPU, N, 1024);
		//Print<uint32_t>(h_OutputGPU, N);

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
		free(h_OutputCPU);
		free(h_OutputGPU);
		printf(" ...Results %s\n\n", (flag == 1) ? "Match" : "DON'T Match !!!");
	}
}