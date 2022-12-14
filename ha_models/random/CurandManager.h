#pragma once

#include "../cuda_kernels/kernels.cuh"
#include "../Tensor.hpp"

#include <vector>
#include <thread>

struct CurandManager {

	cudaStream_t randStream;
	curandState_t* curandStatePool = nullptr;
	size_t poolSize = 0;
	unsigned long seed = 0;
	
	CurandManager();
	CurandManager(size_t statePoolSize, unsigned long seed);

	void randomizeTensorUniform(Tensor_DEVICE t, size_t size, float low, float high);
	void rndOffsetTensorUniform(Tensor_DEVICE t, size_t size, float prob, float low, float high, float zprob);
	
	void randomizeTensorUniform(Tensor& t, float low, float high);
	void rndOffsetTensorUniform(Tensor& t, float prob, float low, float high);
	void rndOffsetTensorUniform(Tensor& t, float prob, float low, float high, float zprob);
	void free();
};