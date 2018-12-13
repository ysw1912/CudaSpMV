#include "pagerank.h"

// 设置纹理对象属性（float型重载）
void setResDesc(cudaResourceDesc &resDesc, float *d_error) {
	resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
}
// 设置纹理对象属性（double型重载）
void setResDesc(cudaResourceDesc &resDesc, double *d_error) {
	resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);
}

