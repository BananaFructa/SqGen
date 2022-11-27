#include "DenseLayer.hpp"
#include "Utils.hpp"

DenseLayer::DenseLayer(size_t inputSize, size_t size, Activation activation) {
	this->size = size;
	this->activation = activation;
	this->inputSize = inputSize;
}

void DenseLayer::free() {
	layer.free();
	biases.free();
	weights.free();
}

size_t DenseLayer::getInputSize() {
	return inputSize;
}

size_t DenseLayer::getOutputSize() {
	return size;
}

unsigned short DenseLayer::stepCount() {
	return 3 + 2 * (activation == Activation::SOFTMAX);
}

unsigned short DenseLayer::stepAsync(Tensor& input) {

	// Only use as much memory space as needed (equal to the batch size of the input)
	Tensor slicedLayer = layer.slice(0, input.size.getDimSize(input.size.dim - 1));

	switch (step++) {
		case 0:
			slicedLayer = input * weights;
			break;
		case 1:
			slicedLayer = slicedLayer + biases;
			break;
		case 2:
			if (activation != Activation::SOFTMAX) {
				slicedLayer.functionPass(activationToKernelFunc(activation));
				step = 0;
			}
			else {
				slicedLayer.functionPass(Func::KERNEL_EXP);
			}
			break;
		case 3:
			slicedLayer.sumAllElementsAcrossDim(auxSumMem);
			break;
		case 4:
			slicedLayer.normalizeAcrossDim(auxSumMem);
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
	auxSumMem.free();
	layer.init(Size((size_t)3, (size_t)1, size, newSize));
	if (activation == Activation::SOFTMAX) auxSumMem.init(Size((size_t)1, newSize));
}

void DenseLayer::rndParams(CurandManager& curandManager) {
	if (weights.size.size == 0) {
		weights.init(Size((size_t)2, inputSize, size));
		biases.init(Size((size_t)2, (size_t)1, size));
	}
	curandManager.randomizeTensorUniform(weights, -1, 1);
	curandManager.randomizeTensorUniform(biases, -1, 1);
}

size_t DenseLayer::loadParams(Tensor params[]) {
	weights = params[0];
	biases = params[1];
	return 2;
}
