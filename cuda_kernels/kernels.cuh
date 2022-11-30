#ifndef TENSOR_KERNELS
#define TENSOR_KERNELS

#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"

#ifndef TENSOR_TYPE
#define TENSOR_TYPE float
#endif

#define NULL_TENSOR 0
#define NULL_TENSOR_MAP 0
#define DEFAULT_STREAM 0

typedef TENSOR_TYPE* Tensor_DEVICE;
typedef TENSOR_TYPE** TensorMap_DEVICE;
typedef TENSOR_TYPE* Scalar;
typedef TENSOR_TYPE* Tensor_HOST;
typedef TENSOR_TYPE** TensorMap_HOST;

enum Func {
	KERNEL_ReLU,
	KERNEL_SIGMOID,
	KERNEL_TANH,
	KERNEL_EXP
};

struct AllocRes {
	Tensor_DEVICE data;
	TensorMap_DEVICE map;
};

void bindStream(cudaStream_t* stream);
void createStream(cudaStream_t* stream);
void destroyStream(cudaStream_t* stream);

// Memory managment
AllocRes allocateTensor(size_t size, size_t mapSize);
void copyTensorFromDevice(Tensor_HOST tHost, Tensor_DEVICE t, size_t size);
void copyTensorFromHost(Tensor_HOST tHost, Tensor_DEVICE t, size_t size);
void copyMapFromHost(TensorMap_DEVICE mHost, TensorMap_DEVICE m, size_t size);
void freeCudaMem(void* t);

void gpuSync();
void gpuSyncStream(cudaStream_t* stream);

namespace CudaKernels {

	// Tensor manipulation
	void normalizeTensor(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize);
	void normalizeTensorMapped(TensorMap_DEVICE m, Tensor_DEVICE sum, size_t poolSize, size_t elemSize, size_t blockSize, size_t allignOffset);

	void funcPass(Tensor_DEVICE t, Func f, size_t size);
	void funcPassMapped(TensorMap_DEVICE m, size_t blockSize, size_t allignOffset, size_t size, Func f);

	void sumTensor(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize);
	void sumTensorMapped(TensorMap_DEVICE m, Tensor_DEVICE sum, size_t poolSize, size_t elemSize, size_t blockSize, size_t allignOffset);

	void addTensor(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t elemSize1, size_t elemSize2, int operand);
	void addTensorMapped(TensorMap_DEVICE mapT, TensorMap_DEVICE map1, TensorMap_DEVICE map2, size_t elemSize1, size_t elemSize2, size_t blockSizeT, size_t blockSize1, size_t blockSize2, size_t allignOffsetT, size_t allignOffset1, size_t allignOffset2, int operand);
	
	void hadamardTensor(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t elemSize1, size_t elemSize2, int operand);
	void hadamardTensorMapped(TensorMap_DEVICE mapT, TensorMap_DEVICE map1, TensorMap_DEVICE map2, size_t elemSize1, size_t elemSize2, size_t blockSizeT, size_t blockSize1, size_t blockSize2, size_t allignOffsetT, size_t allignOffset1, size_t allignOffset2, int operand);
	
	void mulTensor2D(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t poolSize1, size_t poolSize2, size_t l, size_t cl, size_t c, int operand);
	void mulTensorMapped2D(TensorMap_DEVICE tTarget, TensorMap_DEVICE tSource1, TensorMap_DEVICE tSource2, size_t poolSize1, size_t poolSize2, size_t l, size_t cl, size_t c, size_t blockSizeT, size_t blockSize1, size_t blockSize2, size_t allignOffsetT, size_t allignOffset1, size_t allignOffset2, int operand);

	// curand
	void curandStateAlloc(curandState_t* state, size_t size, unsigned long seed);
	void randomizeTensorUniform(curandState_t* state, Tensor_DEVICE t, size_t size, float lowerRange, float higherRange);
	void rndOffsetTensorUniform(curandState_t* state, Tensor_DEVICE t, size_t size, float prob, float lowerRange, float higherRange);

}

#endif