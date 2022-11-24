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

unsigned short DenseLayer::stepAsync(TensorPool2D& input) {
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

TensorPool2D& DenseLayer::getValue() {
	return layer;
}

void DenseLayer::setPool(size_t newSize) {
	layer.free();
	layer.init(newSize, 1, size);
}

void DenseLayer::rndParams() {
	weights.free();
	biases.free();
	// TODO: implementation
}

void DenseLayer::loadParams(Tensor2D params[]) {
	weights = params[0];
	biases = params[1];
}
