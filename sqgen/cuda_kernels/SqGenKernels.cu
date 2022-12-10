#include "SqGenKernels.cuh"

#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"

__global__ void processSIEInputs_kernel(
	TensorMap_DEVICE specieSignalMap,
	Array_DEVICE<size_t> xPositionSet,
	Array_DEVICE<size_t> yPositionSet,
	Tensor_DEVICE inputPool,
	size_t viewRange,
	size_t mapSize,
	size_t signalSize,
	size_t poolSize
) {
	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < poolSize) {
		size_t inputIndex = i / poolSize;
		size_t x = xPositionSet[inputIndex];
		size_t y = yPositionSet[inputIndex];
	}
}


void SqGenKernels::processSIEInputs(
	TensorMap_DEVICE specieSignalMap,
	Array_DEVICE<size_t> xPositionSet,
	Array_DEVICE<size_t> yPositionSet,
	Tensor_DEVICE inputPool,
	size_t viewRange,
	size_t mapSize,
	size_t signalSize,
	size_t poolSize
) {

}