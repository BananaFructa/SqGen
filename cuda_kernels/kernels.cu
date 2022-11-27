#include "kernels.cuh"

#include <math.h>

#include <stdio.h>

#define MAX(a,b) (a > b ? a : b)

cudaStream_t *currentStream = NULL;

__global__ void addTensor_kernel(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t elemSize1, size_t elemSize2, int operand) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t maxSize = max(elemSize1, elemSize2);
	if (i < maxSize) {
		tTarget[i] = tSource1[i % elemSize1] + tSource2[i % elemSize2] + operand * tTarget[i];
	}
}

__global__ void hadamardTensor_kernel(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t elemSize1, size_t elemSize2, int operand) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t maxSize = max(elemSize1, elemSize2);
	if (i < maxSize) {
		tTarget[i] = tSource1[i % elemSize1] * tSource2[i % elemSize2] + operand * tTarget[i];
	}
}

// documentation just for this since is more important
/*
* @param tTarget = The tensor pool in which the results will be stored
* @param tSource1 = First tensor pool in the multiplication operation
* @param tSource2 = Second tensor pool in the multiplication operation
* @param poolSize = The tensor pool size
* @param prodLc = The product between the lines of the first and the columns of the second tensor
* @param l = The number of lines of all the tenors in the first pool
* @param cl = The number of lines/columns of all the tensors from the first/second pool
* @param c = The number of columns of all the tensors in the second pool
* @param single = True if tSource2 is a single tensor and every tensor from tSource1 should be multiplied with it
*/
__global__ void mulTensor_kernel(Tensor_DEVICE tTarget,
								 Tensor_DEVICE tSource1,
	                             Tensor_DEVICE tSource2,
	                             size_t poolSize1,
								 size_t poolSize2,
								 size_t prodLc,
	                             size_t l,
	                             size_t cl,
	                             size_t c,
								 int operand
) {
	size_t t = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tensorStep = t % prodLc;
	size_t poolId = t / prodLc;

	if (poolId < max(poolSize1, poolSize2)) {

		size_t line = tensorStep / c;
		size_t column = tensorStep % c;
		TENSOR_TYPE sum = 0;

		for (size_t i = 0; i < cl; i++) {
			sum += tSource1[(poolId % poolSize1) * prodLc + line + i * l] * tSource2[(poolId % poolSize2) * prodLc + i + column * cl];
		}

		size_t targetId = poolId * prodLc + line + column * l;
		tTarget[targetId] = sum + operand * tTarget[targetId];

	}
}

__global__ void funcPassReLU_kernel(Tensor_DEVICE t, Func f, size_t size) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		bool p = t[i] > 0;
		t[i] = t[i] * p + t[i] * 0.1f * !p;
	}
}

__global__ void funcPassSigmoid_kernel(Tensor_DEVICE t, Func f, size_t size) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		t[i] = 1.0f / (1.0f + expf(-(float)t[i]));
	}
}

__global__ void funcPassTanh_kernel(Tensor_DEVICE t, Func f, size_t size) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		t[i] = tanhf((float)t[i]);
	}
}

__global__ void funcPassExp_kernel(Tensor_DEVICE t, Func f, size_t size) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		t[i] = expf((float)t[i]);
	}
}

__global__ void normalizeTensor_kernel(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tensorStep = i % elemSize;
	size_t poolId = i / elemSize;
	size_t tensorNumber = poolId * elemSize;
	if (poolId < poolSize) {
		t[tensorNumber + tensorStep] /= sum[tensorNumber];
	}
}

__global__ void sumTensor_kernel(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	size_t tensorStep = i % elemSize;
	size_t poolId = i / elemSize;
	size_t tensorNumber = (i / elemSize) * elemSize;
	if (poolId < poolSize) {
		atomicAdd(&sum[tensorNumber], t[tensorNumber + tensorStep]);
	}
}

__global__ void curandInit_kernel(curandState_t* state, size_t size, unsigned long seed) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		curand_init(seed, i, 0, &state[i]);
	}
}

__global__ void randomizeTensorUniform_kernel(curandState_t* state, Tensor_DEVICE t, size_t size, float low, float absoluteDifference) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		t[i] = curand_uniform(&state[i]) * absoluteDifference + low;
	}
}

__global__ void rndOffsetTensorUniform_kernel(curandState_t* state, Tensor_DEVICE t, size_t size, float prob, float low, float absoluteDifference) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size && curand_uniform(&state[i]) <= prob) {
		t[i] += curand_uniform(&state[i]) * absoluteDifference + low;
	}
}

Tensor_DEVICE allocateTensor(size_t size) {
	Tensor_DEVICE tensor;
	cudaMalloc(&tensor, size * sizeof(TENSOR_TYPE));
	return tensor;
}

void bindTensor(cudaStream_t *stream) {
	currentStream = stream;
}

void freeTensor(Tensor_DEVICE t) {
	cudaFree(t);
}

void copyTensorFromDevice(Tensor_HOST tHost,Tensor_DEVICE t, size_t size) {
	cudaMemcpy(tHost, t, size * sizeof(TENSOR_TYPE), cudaMemcpyDeviceToHost);
}

void copyTensorFromHost(Tensor_HOST tHost, Tensor_DEVICE t, size_t size) {
	cudaMemcpy(t, tHost, size * sizeof(TENSOR_TYPE), cudaMemcpyHostToDevice);
}

void CudaKernels::addTensor(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2,size_t elemSize1,size_t elemSize2,int operand) {

	dim3 threadSize(256);
	dim3 blockSize((MAX(elemSize1,elemSize2) + threadSize.x - 1) / threadSize.x);

	addTensor_kernel <<< blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >>> (tTarget, tSource1, tSource2, elemSize1, elemSize2,operand);
}

void CudaKernels::hadamardTensor(Tensor_DEVICE tTarget, Tensor_DEVICE tSource1, Tensor_DEVICE tSource2, size_t elemSize1, size_t elemSize2, int operand) {

	dim3 threadSize(256);
	dim3 blockSize((MAX(elemSize1, elemSize2) + threadSize.x - 1) / threadSize.x);

	hadamardTensor_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (tTarget, tSource1, tSource2, elemSize1, elemSize2,operand);
}

void CudaKernels::funcPass(Tensor_DEVICE t, Func f, size_t size) {

	dim3 threadSize(256);
	dim3 blockSize((size + threadSize.x - 1) / threadSize.x);

	switch (f) {
		case KERNEL_ReLU:
			funcPassReLU_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (t, f, size);
			break;
		case KERNEL_SIGMOID:
			funcPassSigmoid_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (t, f, size);
			break;
		case KERNEL_TANH:
			funcPassTanh_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (t, f, size);
			break;
		case KERNEL_EXP:
			funcPassExp_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (t, f, size);
			break;
	}
}

void CudaKernels::normalizeTensor(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize) {

	dim3 threadSize(256);
	dim3 blockSize((poolSize * elemSize + threadSize.x - 1) / threadSize.x);

	normalizeTensor_kernel <<< blockSize,threadSize, 0, (currentStream ? *currentStream : 0) >>> (t,sum,poolSize,elemSize);
}

void CudaKernels::sumTensor(Tensor_DEVICE t, Tensor_DEVICE sum, size_t poolSize, size_t elemSize) {
	dim3 threadSize(256);
	dim3 blockSize((poolSize * elemSize + threadSize.x - 1) / threadSize.x);

	sumTensor_kernel <<< blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >>> (t, sum, poolSize, elemSize);
}

void CudaKernels::mulTensor2D(Tensor_DEVICE tTarget,Tensor_DEVICE tSource1, Tensor_DEVICE tSource2,size_t poolSize1,size_t poolSize2, size_t l, size_t cl , size_t c,int operand) {
	size_t processCountPerTensor = l * c;
	
	dim3 threadSize(256);
	dim3 blockSize((processCountPerTensor * MAX(poolSize1,poolSize2) + threadSize.x - 1) / threadSize.x);
	
	mulTensor_kernel <<< blockSize,threadSize,0,(currentStream ? *currentStream : 0) >>> (tTarget, tSource1, tSource2, poolSize1,poolSize2,processCountPerTensor, l, cl, c,operand);
}

void CudaKernels::curandStateAlloc(curandState_t* state, size_t size, unsigned long seed) {
	
	dim3 threadSize(256);
	dim3 blockSize((size + threadSize.x - 1) / threadSize.x);

	curandInit_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (state, size, seed);
}

void CudaKernels::randomizeTensorUniform(curandState_t* state, Tensor_DEVICE t, size_t size, float lowerRange, float higherRange) {
	dim3 threadSize(256);
	dim3 blockSize((size + threadSize.x - 1) / threadSize.x);
	
	randomizeTensorUniform_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (state, t, size, lowerRange, fabsf(lowerRange - higherRange));
}

void CudaKernels::rndOffsetTensorUniform(curandState_t* state, Tensor_DEVICE t, size_t size, float prob, float lowerRange, float higherRange) {
	dim3 threadSize(256);
	dim3 blockSize((size + threadSize.x - 1) / threadSize.x);

	rndOffsetTensorUniform_kernel << < blockSize, threadSize, 0, (currentStream ? *currentStream : 0) >> > (state, t, size, prob, lowerRange, fabsf(lowerRange - higherRange));
}

void gpuSync() {
	cudaDeviceSynchronize();
}

void gpuSyncStream(cudaStream_t* stream) {
	cudaStreamSynchronize(*stream);
}

void bindStream(cudaStream_t* stream) {
	currentStream = stream;
}

void createStream(cudaStream_t* stream) {
	cudaStreamCreate(stream);
}

void destroyStream(cudaStream_t* stream) {
	cudaStreamDestroy(*stream);
}
