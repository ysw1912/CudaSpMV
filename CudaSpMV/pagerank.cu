#include "pagerank.h"

// ��������������ԣ�float�����أ�
void setResDesc(cudaResourceDesc &resDesc, float *d_error) {
	resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
}
// ��������������ԣ�double�����أ�
void setResDesc(cudaResourceDesc &resDesc, double *d_error) {
	resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
}

