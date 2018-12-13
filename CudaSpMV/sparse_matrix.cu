#include "sparse_matrix.h"

__host__ std::ostream & operator<<(std::ostream& out, const rowOfBrc& obj)
{
	out << "row = " << obj.row << ", nnz = " << obj.nnz << endl;
	return out;
}
//
//__host__ __device__ bool rowOfBrc::operator<(const rowOfBrc &rhs) const
//{
//	return this->nnz > rhs.nnz;
//}
