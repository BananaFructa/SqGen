#include "CurandManager.h"

CurandManager::CurandManager() {
}

CurandManager::CurandManager(size_t statePoolSize,unsigned long seed) {
	this->poolSize = statePoolSize;
	this->seed = seed;
	createStream(&randStream);
	cudaMalloc(&curandStatePool, poolSize * sizeof(curandState_t));

	bindStream(&randStream);

	CudaKernels::curandStateAlloc(curandStatePool, poolSize, seed);

	bindStream(DEFAULT_STREAM);
}

void CurandManager::randomizeTensorUniform(Tensor_DEVICE t, size_t size, float low, float high) {
	gpuSyncStream(&randStream);
	size_t seg = size / poolSize;
	size_t begin = 0;
	bindStream(&randStream);
	while (size >= poolSize) {
		CudaKernels::randomizeTensorUniform(curandStatePool, t + begin, poolSize, low, high);
		gpuSyncStream(&randStream);
		size -= poolSize;
		begin += poolSize;
	}
	if (size != 0)
	CudaKernels::randomizeTensorUniform(curandStatePool, t + begin, size, low, high);
	bindStream(DEFAULT_STREAM);
}

void CurandManager::rndOffsetTensorUniform(Tensor_DEVICE t, size_t size, float prob, float low, float high) {
	gpuSyncStream(&randStream);
	size_t seg = size / poolSize;
	size_t begin = 0;
	bindStream(&randStream);
	while (size >= poolSize) {
		CudaKernels::rndOffsetTensorUniform(curandStatePool, t + begin, poolSize, prob, low, high);
		gpuSyncStream(&randStream);
		size -= poolSize;
		begin += poolSize;
	}
	if (size != 0)
	CudaKernels::rndOffsetTensorUniform(curandStatePool, t + begin, size, prob, low, high);
	bindStream(DEFAULT_STREAM);
}

void CurandManager::randomizeTensorUniform(Tensor& t, float low, float high) {
	randomizeTensorUniform(t.getGpuDataPointer(), t.size.size, low, high);
}

void CurandManager::rndOffsetTensorUniform(Tensor& t, float prob, float low, float high) {
	rndOffsetTensorUniform(t.getGpuDataPointer(), t.size.size, prob, low, high);
}

void CurandManager::free() {
	cudaFree(curandStatePool);
}
