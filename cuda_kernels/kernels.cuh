#ifndef TENSOR_KERNELS
#define TENSOR_KERNELS

#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"

#define TENSOR_TYPE float
#define NULL_TENSOR 0
#define DEFAULT_STREAM 0

typedef TENSOR_TYPE* Tensor_DEVICE;
typedef TENSOR_TYPE* Scalar;
typedef TENSOR_TYPE* Tensor_HOST;

enum Func {
	KERNEL_ReLU,
	KERNEL_SIGMOID,
	KERNEL_TANH,
	KERNEL_EXP
};

void bindStream(cudaStream_t* stream);
void createStream(cudaStream_t* stream);
void destroyStream(cudaStream_t* stream);

// Memory managment
Tensor_DEVICE allocateTensor(size_t size);
void copyTensorFromDevice(Tensor_HOST tHost, Tensor_DEVICE t, size_t size);
void copyTensorFromHost(Tensor_HOST tHost, Tensor_DEVICE t, size_t size);
void freeTensor(Tensor_DEVICE t);

void gpuSync();
void gpuSyncStream(cudaStream_t* stream);

namespace CudaKernels {

	// Tensor manipulation
	void normalizeTensor(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize);
	void funcPass(Tensor_DEVICE t, Func f, size_t size);

	void sumTensor(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize);
	void addTensor(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t elemSize1, size_t elemSize2, int operand);
	void hadamardTensor(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t elemSize1, size_t elemSize2, int operand);
	void mulTensor2D(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t poolSize1, size_t poolSize2, size_t l, size_t cl, size_t c, int operand);

	// curand
	void curandStateAlloc(curandState_t* state, size_t size, unsigned long seed);
	void randomizeTensorUniform(curandState_t* state, Tensor_DEVICE t, size_t size, float lowerRange, float higherRange);
	void rndOffsetTensorUniform(curandState_t* state, Tensor_DEVICE t, size_t size, float prob, float lowerRange, float higherRange);

}

#endif