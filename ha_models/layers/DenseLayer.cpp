#include "DenseLayer.hpp"

DenseLayer::DenseLayer(size_t poolSize, size_t inputSize, size_t size, Func activation) {
	this->size = size;
	this->activation = activation;

	layer.init(poolSize, 1, size);
	weights.init(inputSize, size);
	biases.init(1, size);
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

void DenseLayer::resizePool(size_t newSize) {
	layer.free();
	layer.init(newSize, 1, size);
}
