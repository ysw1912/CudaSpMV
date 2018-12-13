#include "cpu_helper.h"

void Swap(void *vp1, void *vp2, int sizeBytes) {
	char *buffer = (char *)malloc(sizeBytes);
	if (buffer) {
		memcpy(buffer, vp1, sizeBytes);
		memcpy(vp1, vp2, sizeBytes);
		memcpy(vp2, buffer, sizeBytes);
		free(buffer);
	}
}