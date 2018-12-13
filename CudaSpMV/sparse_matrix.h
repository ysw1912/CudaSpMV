#ifndef __SPARSE_MATRIX__
#define __SPARSE_MATRIX__

#include "mmio.h"
#include "profiler.h"
#include "cpu_helper.h"
#include "gpu_helper.h"

template <class ValueType>
struct triple
{
	uint32_t row;
	uint32_t col;
	ValueType value;

	triple(uint32_t _row, uint32_t _col, ValueType _value = 0)
	: row(_row), col(_col), value(_value) {}

	inline bool operator<(const triple& c) const
	{ return row < c.row || (row == c.row && col < c.col); }
};

class WebMatrix
{
public:
	size_t num_vertices;
	size_t num_edges;

	/* 构造函数 */
	WebMatrix(): num_vertices(0), num_edges(0) {}
	WebMatrix(size_t vertices, size_t edges) : num_vertices(vertices), num_edges(edges) {}

	template <class Matrix>
	WebMatrix(const Matrix &m): num_vertices(m.num_vertices), num_edges(m.num_edges) {}
	
	void resize(size_t vertices, size_t edges)
	{
		num_vertices = vertices;
		num_edges = edges;
	}
};

template <class ValueType>
class CooMatrix : public WebMatrix
{
private:
	using Parent = WebMatrix;
	enum FILE_EXT {CSV, MTX};
	int file_ext;
public:
	vector<uint32_t> row;
	vector<uint32_t> col;
	vector<ValueType> value;

	/* 构造函数 */
	CooMatrix() {}

	CooMatrix(size_t num_vertices, size_t num_edges): Parent(num_vertices, num_edges) {}

	// 读取CSV/Matrix Market文件
	void ReadFileSetValue(const string &filename, bool isDirected, ValueType v = 1.0);

	// 读取CSV/Matrix Market文件，包含value信息
	void ReadFileContainValue(const string &filename, bool isDirected);

	// COO矩阵排序函数
	void sort()
	{
		vector<triple<ValueType>> copy;
		for (size_t i = 0; i < num_edges; i++) {
			copy.emplace_back(row[i], col[i], value[i]);
		}
		std::stable_sort(copy.begin(), copy.end());
		for (size_t i = 0; i < num_edges; i++) {
			row[i] = copy[i].row;
			col[i] = copy[i].col;
			value[i] = copy[i].value;
		}
	}

	// 获取每行非零元素个数并排序
	void showNnz()
	{
		uint32_t *nnz = new uint32_t[num_vertices];
		memset(nnz, 0, num_vertices * sizeof(uint32_t));
		for (uint32_t i = 0; i < num_edges; i++) {
			nnz[row[i]]++;
		}
		std::stable_sort(nnz, nnz + num_vertices);
		std::ofstream ofstrm;
		string resPath = file_path;
		resPath.insert(resPath.size() - 4, "_NNZ");
		ofstrm.open(resPath, ios::out | ios::trunc);       // 文件不存在则建立，存在则覆盖
		for (int i = (int)num_vertices - 1; i >= 0; i--) {
			ofstrm << nnz[i] << endl;
		}
		ofstrm.close();
	}

	// 转置
	void transpose()
	{
		std::swap(row, col);
		sort();
	}

	// 打印COO矩阵
	void Show() const
	{
		size_t n = MIN(row.size(), 15);
		cout << "row:   "; Print<uint32_t>(row.data(), n);
		cout << "col:   "; Print<uint32_t>(col.data(), n);
		cout << "value: "; Print<ValueType>(value.data(), n);
	}
};

template <typename ValueType>
class CsrMatrix : public WebMatrix
{
private:
	using Parent = WebMatrix;
public:
	vector<uint32_t> rowPtr;
	vector<uint32_t> col;
	vector<ValueType> value;

	CsrMatrix() {}
	CsrMatrix(size_t num_vertices, size_t num_edges) : Parent(num_vertices, num_edges) {}

	// 用COO矩阵构造CSR矩阵
	template <typename CooValueType>
	CsrMatrix(const CooMatrix<CooValueType> &coo);

	// 打印CSR矩阵
	void Show() const
	{
		size_t n = MIN(col.size(), 15);
		printf("rowPtr: "); Print<uint32_t>(rowPtr.data(), MIN(rowPtr.size(), 15));
		printf("col:    "); Print<uint32_t>(col.data(), n);
		printf("value:  "); Print<ValueType>(value.data(), n);
	}
};

// BRC矩阵的一行
struct rowOfBrc
{
	uint32_t row;	// 行索引，即矩阵哪一行
	uint32_t nnz;	// 该行nnz，即该行最大索引宽度
	uint32_t currentPtr;	// 当前指针，初始为0

	__host__ __device__ bool operator<(const rowOfBrc &rhs) const
	{
		return this->nnz > rhs.nnz;
	}
};

__host__ std::ostream& operator<<(std::ostream&, const rowOfBrc&);

template <class ValueType>
class BrcMatrix : public WebMatrix
{
private:
	typedef WebMatrix Parent;
public:
	vector<uint32_t> rowPerm;
	vector<uint32_t> col;
	vector<ValueType> value;
	vector<uint32_t> blockPtr;
	vector<uint32_t> block_width;
	uint32_t numBlocks;

	BrcMatrix() {}
	BrcMatrix(size_t num_vertices, size_t num_edges) : Parent(num_vertices, num_edges) {}

	// 用CSR矩阵构造BRC矩阵
	template <typename CsrValueType>
	BrcMatrix(const CsrMatrix<CsrValueType> &csr);

	void Show() const;
};

// BRC Plus Format
template <typename ValueType>
class BrcPMatrix : public WebMatrix
{
private:
	using Parent = WebMatrix;

public:
	vector<uint32_t> rowPerm;
	vector<uint32_t> rowSegLen;	// rowPerm的段长
	vector<uint32_t> col;
	vector<ValueType> value;
	vector<uint32_t> blockPtr;

	BrcPMatrix() {}
	BrcPMatrix(size_t num_vertices, size_t num_edges) : Parent(num_vertices, num_edges) {}

	// 用CSR矩阵构造BrcP矩阵
	template <typename CsrValueType>
	BrcPMatrix(const CsrMatrix<CsrValueType> &csr);

	void Show() const;
};

/* ----------------------------- COO矩阵成员函数 ----------------------------- */

/* 读取CSV/Matrix Market文件并指定value
* filename		-	Matrix Market文件
* isDirected	-	是否有向图
*/
template <typename ValueType>
void CooMatrix<ValueType>::ReadFileSetValue(const string &filename, bool isDirected, ValueType v)
{
	FILE *fp;
	Profiler::Start();
	printf("文件读取...\t\t");
	if (fopen_s(&fp, filename.c_str(), "r") != 0) {	// 打开失败
		Profiler::Finish();
		printf("无法读取文件 \"%s\n\"", filename.c_str());
		exit(-1);
	}
	// 文件扩展名
	string ext = filename.substr(filename.find_last_of('.') + 1);
	if (ext == "csv")
		file_ext = CSV;
	else if (ext == "mtx")
		file_ext = MTX;
	switch (file_ext) {
	case CSV: {
		uint32_t max = 0;
		while (!feof(fp)) {
			uint32_t row_idx, col_idx;
			fscanf_s(fp, "%d,%d\n", &row_idx, &col_idx);
			max = MAX(max, MAX(row_idx, col_idx));
			row.push_back(row_idx);
			col.push_back(col_idx);
			value.push_back(v);
			// 无向图情况
			if (!isDirected) {
				row.push_back(col_idx);
				col.push_back(row_idx);
				value.push_back(v);
			}
		}
		resize(max + 1, row.size());
		break;
	}
	case MTX: {
		MM_typecode matcode;
		if (mm_read_banner(fp, &matcode) != 0) {
			Profiler::Finish();
			printf("Could not process Matrix Market banner.\n");
			exit(1);
		}
		if (mm_read_mtx_crd_size(fp, &num_vertices, &num_vertices, &num_edges) != 0)
			exit(1);
		if (!isDirected)
			num_edges *= 2;
		row.resize(num_edges);
		col.resize(num_edges);
		value.resize(num_edges);
		for (size_t i = 0; i < num_edges; i++) {
			fscanf(fp, "%d %d\n", &row[i], &col[i]);
			value[i] = v;
			/* adjust from 1-based to 0-based */
			row[i]--;
			col[i]--;
			if (!isDirected) {
				++i;
				row[i] = col[i - 1];
				col[i] = row[i - 1];
				value[i] = v;
			}
		}
		break;
	}
	default:
		break;
	}
	if (fp != stdin)
		fclose(fp);
	sort();
	Profiler::Finish();
	printf("%f (s)\n节点数: %zd\n有向边数: %zd\n",
		Profiler::dumpDuration() / CLOCKS_PER_SEC,
		num_vertices, num_edges);
}

/* 读取CSV/Matrix Market文件，包含value信息
* filename		-	Matrix Market文件
* isDirected	-	是否有向图
*/
template <typename ValueType>
void CooMatrix<ValueType>::ReadFileContainValue(const string &filename, bool isDirected)
{
	FILE *fp;
	Profiler::Start();
	printf("文件读取...\t\t");
	if (fopen_s(&fp, filename.c_str(), "r") != 0) {	// 打开失败
		Profiler::Finish();
		printf("无法读取文件 \"%s\n\"", filename.c_str());
		exit(-1);
	}
	// 文件扩展名
	string ext = filename.substr(filename.find_last_of('.') + 1);
	if (ext == "csv")
		file_ext = CSV;
	else if (ext == "mtx")
		file_ext = MTX;
	switch (file_ext) {
	case CSV: {
		uint32_t max = 0;
		while (!feof(fp)) {
			uint32_t row_idx, col_idx;
			ValueType val;
			fscanf_s(fp, "%d,%d,%f\n", &row_idx, &col_idx, &val);
			max = MAX(max, MAX(row_idx, col_idx));
			row.push_back(row_idx);
			col.push_back(col_idx);
			value.push_back(val);
			// 无向图情况
			if (!isDirected) {
				row.push_back(col_idx);
				col.push_back(row_idx);
				value.push_back(val);
			}
		}
		resize(max + 1, row.size());
		break;
	}
	case MTX: {
		MM_typecode matcode;
		if (mm_read_banner(fp, &matcode) != 0) {
			Profiler::Finish();
			printf("Could not process Matrix Market banner.\n");
			exit(1);
		}
		if (mm_read_mtx_crd_size(fp, &num_vertices, &num_vertices, &num_edges) != 0)
			exit(1);
		if (!isDirected)
			num_edges *= 2;
		row.resize(num_edges);
		col.resize(num_edges);
		value.resize(num_edges);
		for (size_t i = 0; i < num_edges; i++) {
			fscanf(fp, "%d %d %f\n", &row[i], &col[i], &value[i]);
			/* adjust from 1-based to 0-based */
			row[i]--;
			col[i]--;
			if (!isDirected) {
				++i;
				row[i] = col[i - 1];
				col[i] = row[i - 1];
				value[i] = value[i - 1];
			}
		}
		break;
	}
	default:
		break;
	}
	if (fp != stdin)
		fclose(fp);
	sort();
	Profiler::Finish();
	printf("%f (s)\n节点数: %zd\n有向边数: %zd\n",
		Profiler::dumpDuration() / CLOCKS_PER_SEC,
		num_vertices, num_edges);
}

/* --------------------------------------------------------------------------- */

/* ----------------------------- CSR矩阵成员函数 ----------------------------- */
// 用COO矩阵构造CSR矩阵
template <typename ValueType>
template <typename CooValueType>
CsrMatrix<ValueType>::CsrMatrix(const CooMatrix<CooValueType> &coo)
{
	resize(coo.num_vertices, coo.num_edges);
	for (size_t i = 0; i < num_edges; i++) {
		col.push_back(coo.col[i]);
		value.push_back(coo.value[i]);
	}
	rowPtr.push_back(0);
	for (size_t i = 1, ptr = 0; i < num_vertices; i++) {
		while (ptr < num_edges && coo.row[ptr] < (uint32_t)i)	ptr++;
		if (ptr == num_edges) {
			rowPtr.push_back((uint32_t)ptr);
			continue;
		}
		if (coo.row[ptr] == i) {
			rowPtr.push_back((uint32_t)ptr);
			ptr++;
		} else if (coo.row[ptr] > i) {
			rowPtr.push_back((uint32_t)ptr);
		}
	}
	rowPtr.push_back((uint32_t)num_edges);
}
/* --------------------------------------------------------------------------- */

/* ----------------------------- BRC矩阵成员函数 ----------------------------- */
// 用CSR矩阵构造BRC矩阵
template <class ValueType>
template <class CsrValueType>
BrcMatrix<ValueType>::BrcMatrix(const CsrMatrix<CsrValueType>& csr)
{
	this->num_vertices = csr.num_vertices;
	this->num_edges = csr.num_edges;

	rowOfBrc* rowVec = new rowOfBrc[num_vertices];
	// vector<rowOfBrc> rowVec(num_vertices);
	uint32_t maxNnz = 0, minNnz = UINT32_MAX;
	for (uint32_t i = 0; i < num_vertices; i++) {
		uint32_t nnz = csr.rowPtr[i + 1] - csr.rowPtr[i];
		maxNnz = MAX(maxNnz, nnz);
		minNnz = MIN(minNnz, nnz);
		rowVec[i] = { i, nnz, 0 };
	}
	/*
	rowOfBrc* d_rowVec;
	checkCudaError(cudaMalloc((void**)&d_rowVec, num_vertices * sizeof(rowOfBrc)));
	checkCudaError(cudaMemcpy(d_rowVec, rowVec, num_vertices * sizeof(rowOfBrc), cudaMemcpyHostToDevice));
	printf("开始排序");
	thrust::stable_sort(d_rowVec, d_rowVec + num_vertices, rowOfBrc_Greater);
	printf("排序结束");
	checkCudaError(cudaMemcpy(rowVec, d_rowVec, num_vertices * sizeof(rowOfBrc), cudaMemcpyDeviceToHost));
	checkCudaError(cudaFree(d_rowVec));
	*/
	std::stable_sort(rowVec, rowVec + num_vertices);

	queue<rowOfBrc> rowQ;
	for (uint32_t i = 0; i < num_vertices && rowVec[i].nnz; i++) {
		rowQ.push(rowVec[i]);
	}

	ValueType mean = ValueType(num_edges) / num_vertices;
	ValueType accum = ValueType(0);
	for (uint32_t i = 0; i < num_vertices; i++) {
		uint32_t t = rowVec[i].nnz;
		accum += (t - mean) * (t - mean);
	}
	ValueType stdev = ValueType(sqrt(accum / num_vertices));
	printf("μ: %f\nσ: %f\n", mean, stdev);

	delete[] rowVec;
	// vector<rowOfBrc>().swap(rowVec);	// 释放内存

	uint32_t B1 = 32, B2 = MIN((uint32_t)(1 * (mean + stdev)), maxNnz);
	//uint32_t B1 = 2, B2 = 2;
	printf("B2: %d\n", B2);
	printf("maxNnz: %d\n", maxNnz);
	printf("minNnz: %d\n", minNnz);
	
	this->numBlocks = 0;
	this->blockPtr.push_back(0);
	while (!rowQ.empty()) {
		ValueType** blockValue = new ValueType*[B1];
		uint32_t** blockCol = new uint32_t*[B1];
		rowOfBrc topRow = rowQ.front();

		// 确定当前块的宽度
		uint32_t blockWidth = MIN(topRow.nnz - topRow.currentPtr, B2);
		this->block_width.push_back(blockWidth);
		// printf("Block - %d Width = %d\n", numBlocks + 1, blockWidth);

		// rowPerm中一次性插入B1个元素
		this->rowPerm.insert(rowPerm.end(), B1, 0);
		for (uint32_t i = 0; i < B1; i++) {
			blockValue[i] = new ValueType[blockWidth];
			blockCol[i] = new uint32_t[blockWidth];
			if (rowQ.empty()) {
				memset(blockCol[i], 0, blockWidth * sizeof(uint32_t));
				memset(blockValue[i], 0, blockWidth * sizeof(ValueType));
			} else {
				rowOfBrc currentRow = rowQ.front();
				rowQ.pop();
				for (uint32_t j = 0; j < blockWidth; ++j) {
					uint32_t rowStart = csr.rowPtr[currentRow.row] + currentRow.currentPtr;
					uint32_t rowEnd = csr.rowPtr[currentRow.row + 1];
					if (j < rowEnd - rowStart) {
						blockValue[i][j] = csr.value[rowStart + j];
						blockCol[i][j] = csr.col[rowStart + j];
					} else {
						blockValue[i][j] = ValueType(0);
						blockCol[i][j] = 0;
					}
				}
				// 按需修改先前插入的元素
				rowPerm[rowPerm.size() - B1 + i] = currentRow.row;
				if (currentRow.nnz - currentRow.currentPtr > B2) {
					currentRow.currentPtr += B2;
					rowQ.push(currentRow);
				}
			}
			//printf("row=%d, ptr=%d\n", currentRow.row, currentRow.currentPtr);
			//cout << "col:   "; Print<uint32_t>(blockCol[i], blockWidth);
		}
		// 将blockValue数据传入value
		//   blockCol  数据传入col
		for (uint32_t j = 0; j < blockWidth; j++) {
			for (uint32_t i = 0; i < B1; i++) {
				value.push_back(blockValue[i][j]);
				col.push_back(blockCol[i][j]);
			}
		}
		/* 释放blockValue、blockCol */
		for (uint32_t i = 0; i < B1; i++) {
			delete[] blockValue[i];
			delete[] blockCol[i];
		}
		delete[] blockValue;
		delete[] blockCol;

		this->blockPtr.push_back(blockPtr[numBlocks] + B1 * block_width[numBlocks]);
		++this->numBlocks;
	}
	// Show();
}

template<class ValueType>
inline void BrcMatrix<ValueType>::Show() const
{
	printf("numBlocks:   %d\n", numBlocks);
	printf("rowPerm:     "); Print<uint32_t>(rowPerm.data(), rowPerm.size());
	printf("col:         "); Print<uint32_t>(col.data(), col.size());
	printf("val:         "); Print<ValueType>(value.data(), value.size());
	printf("block_width: "); Print<uint32_t>(block_width.data(), block_width.size());
	printf("blockPtr:    "); Print<uint32_t>(blockPtr.data(), blockPtr.size());
}
/* --------------------------------------------------------------------------- */

/* ----------------------------- BRCP矩阵成员函数 ----------------------------- */

// 用CSR矩阵构造BRCP矩阵
template <typename ValueType>
template <typename CsrValueType>
BrcPMatrix<ValueType>::BrcPMatrix(const CsrMatrix<CsrValueType> &csr)
{
	// csr.Show();
	this->num_vertices = csr.num_vertices;
	this->num_edges = csr.num_edges;

	thrust::host_vector<rowOfBrc> rowVec(num_vertices);		// 定制的行数组
	uint32_t maxNnz = 0;
	for (uint32_t i = 0; i < num_vertices; i++) {
		uint32_t nnz = csr.rowPtr[i + 1] - csr.rowPtr[i];
		maxNnz = MAX(maxNnz, nnz);
		rowVec[i] = { i, nnz, 0 };
	}
	std::stable_sort(rowVec.begin(), rowVec.end());
	// thrust::device_vector<rowOfBrc> d_rowVec = rowVec;
	// thrust::sort(d_rowVec.begin(), d_rowVec.end());
	// rowVec = d_rowVec;
	// Print<rowOfBrc>(rowVec.data(), rowVec.size());

	ValueType mean = (ValueType)num_edges / num_vertices;
	ValueType accum = (ValueType)0;
	for (uint32_t i = 0; i < num_vertices; i++) {
		uint32_t t = rowVec[i].nnz;
		accum += (t - mean) * (t - mean);
	}
	ValueType stdev = (ValueType)sqrt(accum / num_vertices);
	printf("μ: %f\nσ: %f\n", mean, stdev);

	uint32_t B1 = 32;
	uint32_t B2 = (uint32_t)(1 * round(mean + stdev));
	// uint32_t B2 = maxNnz / 100;
	//uint32_t B1 = 2, B2 = 2;
	printf("B2: %d\n", B2);

	vector<uint32_t> blockWidthArr;
	uint32_t numBlocks = 0;
	blockPtr.push_back(0);
	uint32_t idx = 0;	// rowVec的下标
	while (idx < num_vertices && rowVec[idx].nnz != 0) {
		ValueType** blockValue = new ValueType*[B1];
		uint32_t** blockCol = new uint32_t*[B1];

		// 确定当前块的宽度
		uint32_t blockWidth;
		if (rowVec[idx].currentPtr == 0 && rowVec[idx].nnz > B2 * 64) {
			blockWidth = (uint32_t)ceil(rowVec[idx].nnz / 64);
		}
		else {
			// 该行剩余元素
			uint32_t rest = rowVec[idx].nnz - rowVec[idx].currentPtr;
			if (rest < B2) {
				if (idx + 1 < num_vertices && rowVec[idx + 1].nnz != 0) {	// 由下一行决定
					if (rowVec[idx + 1].nnz <= rest)
						blockWidth = rest;
					else
						blockWidth = MIN(rowVec[idx + 1].nnz, B2);
				}
				else {
					blockWidth = rest;
				}
			}
			else {	// 该行剩余元素≥B2
				blockWidth = B2;
			}
		}
		blockWidthArr.push_back(blockWidth);
		// printf("Block - %d Width = %d\n", numBlocks + 1, blockWidth);

		// rowPerm中一次性插入B1个元素
		rowPerm.insert(rowPerm.end(), B1, 0);
		for (uint32_t i = 0; i < B1; i++) {
			blockValue[i] = new ValueType[blockWidth];	// 当前行的value
			blockCol[i] = new uint32_t[blockWidth];		// 当前行的col
			if (idx == num_vertices) {	// 最后填充0
				memset(blockCol[i], 0, blockWidth * sizeof(uint32_t));
				memset(blockValue[i], 0, blockWidth * sizeof(ValueType));
			}
			else {
				// 按需修改先前插入的元素
				rowPerm[rowPerm.size() - B1 + i] = rowVec[idx].row;

				uint32_t curPtr = rowVec[idx].currentPtr;
				for (uint32_t j = curPtr; j < curPtr + blockWidth; j++) {
					uint32_t rowStart = csr.rowPtr[rowVec[idx].row];
					uint32_t rowEnd = csr.rowPtr[rowVec[idx].row + 1];
					if (j < rowEnd - rowStart) {
						blockValue[i][j - curPtr] = csr.value[rowStart + j];
						blockCol[i][j - curPtr] = csr.col[rowStart + j];
					}
					else {
						blockValue[i][j - curPtr] = (ValueType)0;
						blockCol[i][j - curPtr] = csr.col[rowEnd - 1];
					}
				}

				// 更新rowVec的idx和currentPtr
				rowVec[idx].currentPtr += blockWidth;
				if (rowVec[idx].currentPtr >= rowVec[idx].nnz) {	 // 该行结束
					idx++;
				}
			}
			// printf("value: "); Print<ValueType>(blockValue[i], blockWidth);
			// printf("col:   "); Print<uint32_t>(blockCol[i], blockWidth);
		}
		
		// 将blockValue数据传入value
		//   blockCol  数据传入col
		for (uint32_t j = 0; j < blockWidth; j++) {
			for (uint32_t i = 0; i < B1; i++) {
				value.push_back(blockValue[i][j]);
				col.push_back(blockCol[i][j]);
			}
		}
		// 释放blockValue、blockCol
		for (uint32_t i = 0; i < B1; i++) {
			delete[] blockValue[i];
			delete[] blockCol[i];
		}
		delete[] blockValue;
		delete[] blockCol;

		// blockPtr
		blockPtr.push_back(blockPtr[numBlocks] + B1 * blockWidthArr[numBlocks]);

		// rowSegLen
		vector<uint32_t> segLen(B1);
		uint32_t count = 1;
		for (int i = B1 - 1; i >= 0; i--) {
			if (i == 0 || rowPerm[numBlocks * B1 + i - 1] != rowPerm[numBlocks * B1 + i]) {
				segLen[i] = count;
				count = 1;
			}
			else {
				segLen[i] = 0;
				count++;
			}
		}
		rowSegLen.insert(rowSegLen.end(), segLen.begin(), segLen.end());

		numBlocks++;
	}
	// Show();
}

template <typename ValueType>
inline void BrcPMatrix<ValueType>::Show() const
{
	printf("numBlocks:  %d\n", blockPtr.size() - 1);
	printf("rowPerm:    "); Print<uint32_t>(rowPerm.data(), rowPerm.size());
	printf("rowSegLen:  "); Print<uint32_t>(rowSegLen.data(), rowSegLen.size());
	printf("col:        "); Print<uint32_t>(col.data(), col.size());
	printf("value:      "); Print<ValueType>(value.data(), value.size());
	printf("blockPtr:   "); Print<uint32_t>(blockPtr.data(), blockPtr.size());
}
/* --------------------------------------------------------------------------- */

#endif