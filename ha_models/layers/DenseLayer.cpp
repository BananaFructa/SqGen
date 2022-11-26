#include "DenseLayer.hpp"

DenseLayer::DenseLayer(size_t inputSize, size_t size, Func activation) {
	this->size = size;
	this->activation = activation;
	this->inputSize = inputSize;
}

void DenseLayer::free() {
	freeLayers();
	biases.free();
	weights.free();
}

void DenseLayer::freeLayers() {
	layer.free();
}

size_t DenseLayer::getInputSize() {
	return inputSize;
}

size_t DenseLayer::getOutputSize() {
	return size;
}

unsigned short DenseLayer::stepCount() {
	return DENSE_LAYER_STEP_COUNT;
}

unsigned short DenseLayer::stepAsync(Tensor& input) {
	switch (step++) {
		case 0:
			layer = input * weights;
			break;
		case 1:
			layer = layer + biases;
			break;
		case 2:
			layer.functionPass(activation);
			step = 0;
			break;
	}
	return step;
}

Tensor& DenseLayer::getValue() {
	return layer;
}

void DenseLayer::setPool(size_t newSize) {
	layer.free();
	layer.init(Size(3, 1, size, newSize));
}

void DenseLayer::rndParams(CurandManager& curandManager) {
	if (weights.size.size == 0) {
		weights.init(Size(2, inputSize, size));
		biases.init(Size(2, 1, size));
	}
	curandManager.randomizeTensorUniform(weights, -1, 1);
	curandManager.randomizeTensorUniform(biases, -1, 1);
}

void DenseLayer::loadParams(Tensor params[]) {
	weights = params[0];
	biases = params[1];
}
