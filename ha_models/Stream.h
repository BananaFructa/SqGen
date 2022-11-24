#pragma once

#include "../cuda_kernels/kernels.cuh"

struct Stream {
public:

	Stream() {
		cudaStreamCreate(&stream);
	}

	~Stream() {
		cudaStreamDestroy(stream);
	}

private:

	cudaStream_t stream;

};