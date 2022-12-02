#include "SimpleRecurrentLayer.hpp"
#include "Utils.hpp"

// TODO: load / save state

// Doesn't support SOFTMAX
SimpleRecurrentLayer::SimpleRecurrentLayer(size_t inputSize, size_t size, Activation activation, Activation hiddenActivation) {
	this->inputSize = inputSize;
	this->size = size;
	this->activation = activation;
	this->hiddenActivation = hiddenActivation;
}

void SimpleRecurrentLayer::free() {
	weightsInput.free();
	weightsHiddenPast.free();
	weightsHiddenPresent.free();
	biasesHidden.free();
	biasesOutput.free();
	hiddenLayer.free();
	layer.free();
}

void SimpleRecurrentLayer::setPool(size_t newSize) {
	Size outSize = Size((size_t)3, (size_t)1, size, newSize);
	if (allocInternal) {
		hiddenLayer.init(outSize);
		hiddenLayer.initZero();
		gpuSync();
	}
	layer.init(outSize);
}

size_t SimpleRecurrentLayer::getInputSize() {
	return inputSize;
}

size_t SimpleRecurrentLayer::getOutputSize() {
	return size;
}

unsigned short SimpleRecurrentLayer::stepCount() {
	return 6;
}

unsigned short SimpleRecurrentLayer::stepAsync(Tensor& input) {

	Tensor layerSliced = layer.slice(0, input.size.last());
	Tensor hiddenSliced = hiddenLayer.slice(0, input.size.last());

	switch (step++) {
		case 0:
			hiddenSliced = hiddenSliced % weightsHiddenPast;
			break;
		case 1:
			hiddenSliced += input * weightsInput;
			break;
		case 2:
			hiddenSliced = hiddenSliced + biasesHidden; // 1
			break;
		case 3:
			hiddenSliced.functionPass(activationToKernelFunc(hiddenActivation));
			break;
		case 4:
			layerSliced = hiddenSliced % weightsHiddenPresent;
			break;
		case 5:
			layerSliced = layerSliced + biasesOutput; // 1
			break;
		case 6:
			layerSliced.functionPass(activationToKernelFunc(activation));
			step = 0;
			break;
	}
	return step;
}

void SimpleRecurrentLayer::rndParams(CurandManager& curandManager) {
	if (weightsInput.size.size == 0) {
		weightsInput.init(Size((size_t)2, inputSize, size));
		weightsHiddenPast.init(Size((size_t)2, 1, size));
		weightsHiddenPresent.init(Size((size_t)2, 1, size));
		biasesHidden.init(Size((size_t)2, 1, size));
		biasesOutput.init(Size((size_t)2, 1, size));
	}
	curandManager.randomizeTensorUniform(weightsInput, -1, 1);
	curandManager.randomizeTensorUniform(weightsHiddenPast, -1, 1);
	curandManager.randomizeTensorUniform(weightsHiddenPresent, -1, 1);
	curandManager.randomizeTensorUniform(biasesHidden, -1, 1);
	curandManager.randomizeTensorUniform(biasesOutput, -1, 1);
}

size_t SimpleRecurrentLayer::loadParams(Tensor params[]) {
	weightsInput = params[0];
	weightsHiddenPast = params[1];
	weightsHiddenPresent = params[2];
	biasesHidden = params[3];
	biasesOutput = params[4];
	return 5;
}

size_t SimpleRecurrentLayer::loadState(Tensor state[]) {
	hiddenLayer = state[0];
	return 1;
}

Tensor& SimpleRecurrentLayer::getValue() {
	return layer;
}
