#include "UniquelyMappedTensor.hpp"


UniquelyMappedTensor::UniquelyMappedTensor() : Tensor() {
}


UniquelyMappedTensor::UniquelyMappedTensor(Size size) {
	init(size);
}


void UniquelyMappedTensor::init(Size size) {
	if (!Tensor::gpuTensorData) {
		this->size = size;
		this->mapSize = size.last();
		this->mapBlockSize = size.size / Tensor::mapSize;
		this->mapped = true;
		AllocRes res = allocateTensor(size.size, Tensor::mapSize);
		Tensor::gpuTensorData = res.data;
		Tensor::gpuTensorMap = res.map;

		Tensor::hostMap = new TENSOR_TYPE * [Tensor::mapSize];
		for (int i = 0; i < Tensor::mapSize; i++) {
			Tensor::hostMap[i] = Tensor::gpuTensorData + i * Tensor::mapBlockSize;
		}
		Tensor::syncMap();
	}
}


void UniquelyMappedTensor::swap(size_t a, size_t b) {
	Tensor_DEVICE temp = Tensor::hostMap[a];
	Tensor::hostMap[a] = Tensor::hostMap[b];
	Tensor::hostMap[b] = temp;
}
