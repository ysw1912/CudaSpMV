#include "test.h"

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>

namespace test
{
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

	void test02()
	{
		FILE *f;
		const string filename = "F:\\networks\\matrices\\qcd5_4.mtx";
		if (fopen_s(&f, filename.c_str(), "r") != 0) {
			printf("无法读取文件 \"%s\n\"", filename.c_str());
			exit(1);
		}
		MM_typecode matcode;
		if (mm_read_banner(f, &matcode) != 0) {
			printf("Could not process Matrix Market banner.\n");
			exit(1);
		}

		/* find out size of sparse matrix .... */
		size_t M, N, nz;
		if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
			exit(1);

		/* reseve memory for matrices */
		int* I = (int *)malloc(nz * sizeof(int));
		int* J = (int *)malloc(nz * sizeof(int));
		double *val = (double *)malloc(nz * sizeof(double));

		/* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
		/*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
		/*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
		for (int i = 0; i < nz; i++) {
			fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
			/* adjust from 1-based to 0-based */
			I[i]--;
			J[i]--;
		}

		if (f != stdin) fclose(f);
	}
}