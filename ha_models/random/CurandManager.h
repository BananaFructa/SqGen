#pragma once

#include "../../cuda_kernels/kernels.cuh"

#include <vector>
#include <thread>

struct CurandManager {

	cudaStream_t randStream;
	curandState_t* curandStatePool;
	size_t poolSize = 0;
	unsigned long seed;
	
	CurandManager(size_t statePoolSize, unsigned long seed);

	void randomizeTensorUniform(Tensor t, size_t size, float low, float high);
	void rndOffsetTensorUniform(Tensor t, size_t size, float prob, float low, float high);

	~CurandManager();
};