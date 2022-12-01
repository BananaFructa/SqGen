#include "UniquelyMappedTensor.hpp"

UniquelyMappedTensor::UniquelyMappedTensor() : Tensor() {
}

UniquelyMappedTensor::UniquelyMappedTensor(Size size) {
	init(size);
}

void UniquelyMappedTensor::init(Size size) {
	if (!gpuTensorData) {
		this->size = size;
		this->mapSize = size.last();
		this->mapBlockSize = size.size / mapSize;
		this->mapped = true;
		AllocRes res = allocateTensor(size.size, mapSize);
		gpuTensorData = res.data;
		gpuTensorMap = res.map;

		hostMap = new TENSOR_TYPE * [mapSize];
		for (int i = 0; i < mapSize; i++) {
			hostMap[i] = gpuTensorData + i * mapBlockSize;
		}
		syncMap();
	}
}

void UniquelyMappedTensor::swap(size_t a, size_t b) {
	Tensor_DEVICE temp = hostMap[a];
	hostMap[a] = hostMap[b];
	hostMap[b] = temp;
}
