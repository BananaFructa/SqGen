#include "SqGenKernels.cuh"
#include "../Constant.h"

#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"

// TODO: test all of this modify agent pos directly and check signal

__global__ void processSIEInputs_kernel(
	Array_DEVICE<short> logicMap,
	Array_DEVICE<float> distanceMap,
	TensorMap_DEVICE specieSignalMap, // Tensor which holds the specie signal that is present at each map position (signal size)
	Array_DEVICE<short> xPositionSet, // X positions of the agents
	Array_DEVICE<short> yPositionSet, // Y position of the agents
	Tensor_DEVICE inputPool, // The input layer of the SIE (1, signal size,agent count * 4)
	size_t viewRange,
	size_t mapSize,
	size_t signalSize,
	size_t agentCount
) {
	size_t t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t < agentCount) {

		int cx = xPositionSet[t]; // position x of the agent
		int cy = yPositionSet[t]; // position y of the agent
		size_t indexInModelInput = t * 4; // The index in the SIE input tensor
		
		size_t x, y;

		for (size_t i = 0; i < (viewRange * 2 + 1) * (viewRange * 2 + 1); i++) {
			// first
			x = cx - viewRange + i / (viewRange * 2 + 1);
			y = cy - viewRange + i % (viewRange * 2 + 1);
			x = (x + mapSize) % mapSize;
			y = (y + mapSize) % mapSize;
			short logic = logicMap[i];
			for (size_t j = 0; j < signalSize; j++) {
				inputPool[(indexInModelInput + ((logic & 0b1100) >> 2)) * signalSize * 2 + j] += specieSignalMap[y + x * mapSize][j] * (float)((logic & USE_SECOND) >> 5) * distanceMap[i];
				inputPool[(indexInModelInput + (logic & 0b11)) * signalSize * 2 + j] += specieSignalMap[y + x * mapSize][j] * (float)((logic & USE_FIRST) >> 4) * distanceMap[i];
			}
		}

		for (size_t j = 0; j < signalSize; j++) {
			for (size_t k = 0; k < 4; k++) {
				inputPool[(indexInModelInput + k) * signalSize * 2 + signalSize + j] = specieSignalMap[cy + cx * mapSize][j];
			}
		}

		// Normalization step

		for (size_t j = 0; j < 4; j++) {

			float _max = 0;

			for (size_t i = 0; i < signalSize; i++) {
				//                                                  V here as well
				_max = max(_max, abs(inputPool[(indexInModelInput + j) * signalSize * 2 + i]));
			}

			float biggerThan1 = min(1.0f,(float)(int)_max);

			for (size_t i = 0; i < signalSize; i++) {
				//                             V and here
				inputPool[(indexInModelInput + j) * signalSize * 2 + i] /= ((_max - 1) * biggerThan1 + 1);
			}
		}

	}
}

__global__ void processAPSGInputs_kernel(
	Array_DEVICE<short> logicMap,
	Array_DEVICE<short> xPositionSet,
	Array_DEVICE<short> yPositionSet,
	Tensor_DEVICE SIE_Output,
	Tensor_DEVICE foodMap,
	Array_DEVICE<float> foodLevels,
	Tensor_DEVICE signalMap,
	Tensor_DEVICE inputPool,
	size_t viewRange,
	size_t mapSize,
	size_t agentCount
) {
	size_t t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t < agentCount) {
		int cx = xPositionSet[t];
		int cy = yPositionSet[t];

		// Current agent food
		inputPool[t * 10 + 0] = foodLevels[t] / Constants::FmaximumFood;
		// Food value of the tile
		inputPool[t * 10 + 1] = max(0.0f,
			foodMap[(cy + 1) % mapSize + cx * mapSize]+
			foodMap[(cy - 1) % mapSize + cx * mapSize]+
			foodMap[cy + ((cx + 1) % mapSize) * mapSize]+
			foodMap[cy + ((cx - 1) % mapSize) * mapSize]
		);

		// Copy the visual data from the SIE output
		for (size_t i = 0; i < 4; i++) {
			inputPool[t * 10 + 2 + i] = SIE_Output[t * 4 + i];
		}

		size_t x, y;

		// Similar thing to the SIE kernel but for the normal signal map and no normalization
		for (size_t i = 0; i < (viewRange * 2 + 1) * (viewRange * 2 + 1); i++) {
			// first
			x = (cx-viewRange) + i / (viewRange * 2 + 1);
			y = (cy-viewRange) + i % (viewRange * 2 + 1);
			x = (x + mapSize) % mapSize;
			y = (y + mapSize) % mapSize;
			short logic = logicMap[i];
			inputPool[(t * 10 + 6 + ((logic & 0b1100) >> 2))] += signalMap[y + x * mapSize] * (float)((logic & USE_SECOND) >> 5);
			inputPool[(t * 10 + 6 + (logic & 0b11))] += signalMap[y + x * mapSize] * (float)((logic & USE_FIRST) >> 4);
		}

		for (size_t i = 0; i < 4; i++) {
			inputPool[t * 10 + 6 + i] = max(min(inputPool[t * 10 + 6 + i], -1.0f), 1.0f);
		}

	}
}


void SqGenKernels::processSIEInputs(
	Array_DEVICE<short> logicMap,
	Array_DEVICE<float> distanceMap,
	TensorMap_DEVICE specieSignalMap,
	Array_DEVICE<short> xPositionSet,
	Array_DEVICE<short> yPositionSet,
	Tensor_DEVICE inputPool,
	size_t viewRange,
	size_t mapSize,
	size_t signalSize,
	size_t agentCount
) {
	dim3 threadSize(256);
	dim3 blockSize((agentCount + threadSize.x - 1) / threadSize.x);

	processSIEInputs_kernel <<< blockSize, threadSize>>> (logicMap,distanceMap, specieSignalMap, xPositionSet, yPositionSet, inputPool, viewRange, mapSize, signalSize, agentCount);
}

void SqGenKernels::processAPSGInputs(
	Array_DEVICE<short> logicMap,
	Array_DEVICE<short> xPositionSet,
	Array_DEVICE<short> yPositionSet,
	Tensor_DEVICE SIE_Output,
	Tensor_DEVICE foodMap,
	Array_DEVICE<float> foodLevels,
	Tensor_DEVICE signalMap,
	Tensor_DEVICE inputPool,
	size_t viewRange,
	size_t mapSize,
	size_t agentCount
) {
	dim3 threadSize(256);
	dim3 blockSize((agentCount + threadSize.x - 1) / threadSize.x);

	processAPSGInputs_kernel <<< blockSize, threadSize >>> (logicMap, xPositionSet, yPositionSet, SIE_Output, foodMap, foodLevels, signalMap, inputPool, viewRange, mapSize, agentCount);
}
