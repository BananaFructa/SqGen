#include "ReferenceMappedTensor.hpp"

ReferenceMappedTensor::ReferenceMappedTensor() : Tensor() {
}

ReferenceMappedTensor::ReferenceMappedTensor(Size size) {
	init(size);
}

void ReferenceMappedTensor::init(Size size) {
	if (!gpuTensorMap) {
		this->size = size;
		this->mapSize = size.last();
		this->mapBlockSize = size.size / mapSize;
		this->mapped = true;
		this->referenceOnly = true;
		AllocRes res = allocateTensor(0, mapSize);
		gpuTensorData = res.data;
		gpuTensorMap = res.map;
		hostMap = new TENSOR_TYPE * [mapSize];
		for (int i = 0; i < mapSize; i++) {
			hostMap[i] = NULL_TENSOR;
		}
		syncMap();
	}
}

void ReferenceMappedTensor::setRef(size_t index, Tensor& t) {
	if (!t.mapped) hostMap[index] = t.getGpuDataPointer();
}

void ReferenceMappedTensor::setRef(size_t index, Scalar s) {
	hostMap[index] = s;
}

void ReferenceMappedTensor::swap(size_t a, size_t b) {
	Tensor_DEVICE temp = hostMap[a];
	hostMap[a] = hostMap[b];
	hostMap[b] = temp;
}
