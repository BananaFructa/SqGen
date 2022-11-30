#include "UniquelyMappedTensor.hpp"

UniquelyMappedTensor::UniquelyMappedTensor() : Tensor() {
}

UniquelyMappedTensor::UniquelyMappedTensor(Size size) : Tensor(size) {
}

void UniquelyMappedTensor::init(Size size) {
	if (!gpuTensorData) {
		this->size = size;
		this->mapSize = size.last();
		this->mapBlockSize = size.size / mapSize;
		AllocRes res = allocateTensor(size.size, mapSize);
		gpuTensorData = res.data;
		gpuTensorMap = res.map;

		TENSOR_TYPE** map = new TENSOR_TYPE * [mapSize];
		for (int i = 0; i < mapSize; i++) {
			map[i] = gpuTensorData + i * mapBlockSize;
		}

		setMap((TensorMap_HOST)map);

		delete[] map;

	}
}
