#include "../../ha_models/cuda_kernels/kernels.cuh"

namespace SqGenKernels {

	void processSIEInputs(
		TensorMap_DEVICE specieSignalMap,
		Array_DEVICE<size_t> xPositionSet,
		Array_DEVICE<size_t> yPositionSet,
		Tensor_DEVICE inputPool,
		size_t viewRange,
		size_t mapSize,
		size_t signalSize,
		size_t poolSize
	);

}