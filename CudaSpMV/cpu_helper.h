#ifndef __CPU_HELPER__
#define __CPU_HELPER__

#include <cassert>
#include <cstdint>
#include <ctime>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::queue;
using std::setiosflags;
using std::setw;
using std::ios;
using std::unique_ptr;

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/*
** 输出ep指向的T类型开始的n个元素
** 设show_all为false则只打印前6个以及后4个元素
**/
template <typename T>
void Print(const void* ep, size_t n, bool show_all = true)
{
	printf("{ ");
	if (show_all || n <= 10) {
		for (size_t i = 0; i < n; i++)
			cout << *(static_cast<const T*>(ep) + i) << " ";
	}
	else {
		for (size_t i = 0; i < 6; i++)
			cout << *(static_cast<const T*>(ep) + i) << " ";
		printf("... ");
		for (size_t i = n - 4; i < n; i++)
			cout << *(static_cast<const T*>(ep) + i) << " ";
	}
	printf("}\n");
}

template <typename T>
void PrintByRow(const void* ep, size_t n, size_t stride)
{
	printf("{ ");
	size_t i = 0, row = 0;
	while (true) {
		if (row)	printf("\n\t");
		for (i = row * stride; i < row * stride + 6; i++)
			cout << *(static_cast<const T*>(ep) + i) << "\t";
		printf("...   ");
		size_t bound = MIN((row + 1) * stride, n);
		for (size_t i = bound - 4; i < bound; i++)
			cout << *(static_cast<const T*>(ep) + i) << "\t";
		if (bound == n)	break;
		++row;
	}
	printf("}\n");
}

// 交换vp1和vp2指向地址的sizeBytes
void Swap(void *vp1, void *vp2, int sizeBytes);

// 两向量的平均绝对误差，lhs为真值
template <typename ValueType>
ValueType MeanAbsoluteError(const vector<ValueType>& lhs, const vector<ValueType>& rhs)
{
	size_t n = lhs.size();
	assert(rhs.size() == n);
	ValueType res = 0;
	for (size_t i = 0; i < n; i++) {
		res += fabs(rhs[i] - lhs[i]);
	}
	return res / n;
}

// 两向量的相对误差，lhs为真值
template <typename ValueType>
ValueType RelativeError(const vector<ValueType>& lhs, const vector<ValueType>& rhs)
{
	size_t n = lhs.size();
	assert(rhs.size() == n);
	ValueType res = 0;
	for (size_t i = 0; i < n; i++) {
		res += fabs(rhs[i] - lhs[i]) / lhs[i];
	}
	return res;
}

template <typename ValueType>
void Func(const vector<ValueType>& res,
	const vector<ValueType>& res1,
	const vector<ValueType>& res2)
{
	FILE* fp = fopen("F:\\networks\\func.csv", "w+");
	fprintf_s(fp, "res,res1,res2,res1 - res,RE1,res2 - res,RE2\n");
	for (size_t i = 0; i < res.size(); ++i) {
		fprintf_s(fp, "%f,%f,%f,%f,%f,%f,%f\n",
			res[i], res1[i], res2[i],
			fabs(res1[i] - res[i]),
			fabs(res1[i] - res[i]) / res[i],
			fabs(res2[i] - res[i]),
			fabs(res2[i] - res[i]) / res[i]);
	}
	fclose(fp);
}

#endif