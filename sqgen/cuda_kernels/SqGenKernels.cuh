#include "../../ha_models/cuda_kernels/kernels.cuh"

#define LEFT_FIRST 0b00 
#define RIGHT_FIRST 0b01
#define UP_FIRST 0b10
#define DOWN_FIRST 0b11

#define LEFT_SECOND 0b0000
#define RIGHT_SECOND 0b0100
#define UP_SECOND 0b1000
#define DOWN_SECOND 0b1100

#define USE_FIRST 0b10000
#define USE_SECOND 0b100000

namespace SqGenKernels {

	void processSIEInputs(
		Array_DEVICE<short> logicMap,
		Array_DEVICE<float> distanceMap,
		TensorMap_DEVICE specieSignalMap,
		Array_DEVICE<int> xPositionSet,
		Array_DEVICE<int> yPositionSet,
		Tensor_DEVICE inputPool,
		size_t viewRange,
		size_t mapSize,
		size_t signalSize,
		size_t poolSize
	);

	void processAPSGInputs(
		Array_DEVICE<short> logicMap,
		Array_DEVICE<int> xPositionSet,
		Array_DEVICE<int> yPositionSet,
		Tensor_DEVICE SIE_Output,
		Tensor_DEVICE foodMap,
		Array_DEVICE<float> foodLevels,
		Tensor_DEVICE signalMap,
		Tensor_DEVICE inputPool,
		size_t viewRange,
		size_t mapSize,
		size_t agentCount
	);

}