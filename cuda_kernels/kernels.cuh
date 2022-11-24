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

typedef TENSOR_TYPE* Tensor;
typedef TENSOR_TYPE* Scalar;
typedef TENSOR_TYPE* Tensor_HOST;

enum Func {
	ReLU,
	SIGMOID,
	TANH,
	EXP
};

void bindStream(cudaStream_t* stream);

// Memory managment
Tensor allocateTensor(size_t size);
void copyTensorFromDevice(Tensor_HOST tHost, Tensor t, size_t size);
void copyTensorFromHost(Tensor_HOST tHost, Tensor t, size_t size);
void freeTensor(Tensor t);

// Tensor manipulation
void normalizeTensor(Tensor t, Tensor sum, size_t poolSize, size_t elemSize);
void funcPass(Tensor t, Func f, size_t size);

void sumTensor(Tensor t, Tensor sum,size_t poolSize, size_t elemSize);
void addTensor(Tensor tTarget, Tensor tSource1, Tensor tSource2, size_t elemSize, size_t size, bool single);
void mulTensor2D(Tensor tTarget, Tensor tSource1, Tensor tSource2,size_t poolSize, size_t l, size_t cl, size_t c,bool single);

// curand
void curandStateAlloc(curandState_t** state, size_t size);
void randomizeTensor(curandState_t *satate, Tensor t,size_t size, float propPerElement, float lowerRanger, float higherRange);

void gpuSync();

#endif