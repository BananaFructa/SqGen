#include "kernels.cuh"

#include <math.h>

#include <stdio.h>

cudaStream_t *currentStream = NULL;

__global__ void addTensor_kernel(Tensor tTarget, Tensor tSource1, Tensor tSource2, size_t elemSize, size_t totalSize, bool single) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < totalSize) {
		tTarget[i] = tSource1[i] + tSource2[(single ? i % elemSize : i)];
	}
}

__global__ void mulTensor2D_kernel(Tensor tTarget, // Target pool
								   Tensor tSource1, // Source pool 1
	                               Tensor tSource2, // Source pool 2
	                               size_t poolSize, // Pool size
								   size_t prodLc, // Product of l * c
	                               size_t l, // lines first
	                               size_t cl, // columns first = lines second
	                               size_t c, // columns second
								   bool single
) {
	size_t t = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tensorStep = t % prodLc; // The processing step in the current tensor
	size_t poolId = t / prodLc;
	size_t tensorNumber = (poolId) * prodLc; // The current tensor from the pool

	if (poolId < poolSize) {

		size_t line = tensorStep / c;
		size_t column = tensorStep % c;
		TENSOR_TYPE sum = 0;

		for (size_t i = 0; i < cl; i++) {
			sum += tSource1[tensorNumber + i + line * cl] * tSource2[(single ? 0 :tensorNumber) + column + i * c];
		}

		tTarget[tensorNumber + column + line * c] = sum;

	}
}

__global__ void funcPass_kernel(Tensor t, Func f, size_t size) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		switch (f) {
			case ReLU:
				bool p = t[i] > 0;
				t[i] = t[i] * p + t[i] * 0.1f * !p;
				break;
			case SIGMOID:
				t[i] = 1.0f / (1.0f + expf(-(float)t[i]));
				break;
			case TANH:
				t[i] = tanhf((float)t[i]);
				break;
			case EXP:
				t[i] = expf((float)t[i]);
				break;
		}
	}
}

__global__ void normalizeTensor_kernel(Tensor t, Tensor sum, size_t poolSize, size_t elemSize) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tensorStep = i % elemSize;
	size_t poolId = i / elemSize;
	size_t tensorNumber = poolId * elemSize;
	if (poolId < poolSize) {
		t[tensorNumber + tensorStep] /= sum[tensorNumber];
	}
}

__global__ void sumTensor_kernel(Tensor t, Tensor sum, size_t poolSize, size_t elemSize) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tensorStep = i % elemSize;
	size_t poolId = i / elemSize;
	size_t tensorNumber = (i / elemSize) * elemSize;
	if (poolId < poolSize) {
		atomicAdd(&sum[tensorNumber], t[tensorNumber + tensorStep]);
	}
}

Tensor allocateTensor(size_t size) {
	Tensor tensor;
	cudaMalloc(&tensor, size * sizeof(TENSOR_TYPE));
	return tensor;
}

void bindTensor(cudaStream_t *stream) {
	currentStream = stream;
}

void freeTensor(Tensor t) {
	cudaFree(t);
}

void copyTensorFromDevice(Tensor_HOST tHost,Tensor t, size_t size) {
	cudaMemcpy(tHost, t, size * sizeof(TENSOR_TYPE), cudaMemcpyDeviceToHost);
}

void copyTensorFromHost(Tensor_HOST tHost, Tensor t, size_t size) {
	cudaMemcpy(t, tHost, size * sizeof(TENSOR_TYPE), cudaMemcpyHostToDevice);
}

void addTensor(Tensor tTarget, Tensor tSource1, Tensor tSource2,size_t elemSize, size_t size, bool single) {

	dim3 threadSize(256);
	dim3 blockSize((size + threadSize.x - 1) / threadSize.x);

	addTensor_kernel <<< blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >>> (tTarget, tSource1, tSource2, elemSize, size, single);
}

void funcPass(Tensor t, Func f, size_t size) {

	dim3 threadSize(256);
	dim3 blockSize((size + threadSize.x - 1) / threadSize.x);

	funcPass_kernel <<< blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >>> (t, f, size);
}

void normalizeTensor(Tensor t, Tensor sum, size_t poolSize, size_t elemSize) {

	dim3 threadSize(256);
	dim3 blockSize((poolSize * elemSize + threadSize.x - 1) / threadSize.x);

	normalizeTensor_kernel <<< blockSize,threadSize, 0, (currentStream ? *currentStream : 0) >>> (t,sum,poolSize,elemSize);
}

void sumTensor(Tensor t, Tensor sum, size_t poolSize, size_t elemSize) {
	dim3 threadSize(256);
	dim3 blockSize((poolSize * elemSize + threadSize.x - 1) / threadSize.x);

	sumTensor_kernel <<< blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >>> (t, sum, poolSize, elemSize);
}

void mulTensor2D(Tensor tTarget,Tensor tSource1, Tensor tSource2,size_t poolSize, size_t l, size_t cl , size_t c, bool single) {
	dim3 threadSize(256);
	dim3 blockSize((l * c * poolSize + threadSize.x - 1) / threadSize.x);
	
	mulTensor2D_kernel <<< blockSize,threadSize,0,(currentStream ? *currentStream : 0) >>> (tTarget, tSource1, tSource2, poolSize,l*c, l, cl, c,single);
}

void gpuSync() {
	cudaDeviceSynchronize();
}
