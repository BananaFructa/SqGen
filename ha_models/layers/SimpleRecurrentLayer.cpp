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

Size SimpleRecurrentLayer::getInputSize() {
	return Size((size_t)2,(size_t)1,inputSize);
}

Size SimpleRecurrentLayer::getOutputSize() {
	return Size((size_t)2,(size_t)1,size);
}

unsigned short SimpleRecurrentLayer::stepCount() {
	return 6;
}

unsigned short SimpleRecurrentLayer::stepAsync(Tensor input) {

	Tensor layerSliced = layer.slice(0, input.size.last());
	Tensor hiddenSliced = hiddenLayer.slice(0, input.size.last());

	lastSize = input.size.last();

	switch (step++) {
		case 0:
			hiddenSliced = hiddenSliced % weightsHiddenPast;
			break;
		case 1:
			hiddenSliced += input * weightsInput;
			break;
		case 2:
			hiddenSliced = hiddenSliced + biasesHidden;
			break;
		case 3:
			hiddenSliced.functionPass(activationToKernelFunc(hiddenActivation));
			break;
		case 4:
			layerSliced = hiddenSliced % biasesHidden;
			break;
		case 5:
			layerSliced = layerSliced + biasesOutput;
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
		weightsInput.init(Size((size_t)3, inputSize, size, (size_t)1));
		weightsHiddenPast.init(Size((size_t)3, 1, size, (size_t)1));
		weightsHiddenPresent.init(Size((size_t)3, 1, size, (size_t)1));
		biasesHidden.init(Size((size_t)3, 1, size, (size_t)1));
		biasesOutput.init(Size((size_t)3, 1, size, (size_t)1));
	}
	curandManager.randomizeTensorUniform(weightsInput, -1, 1);
	curandManager.randomizeTensorUniform(weightsHiddenPast, -1, 1);
	curandManager.randomizeTensorUniform(weightsHiddenPresent, -1, 1);
	curandManager.randomizeTensorUniform(biasesHidden, -1, 1);
	curandManager.randomizeTensorUniform(biasesOutput, -1, 1);
}

size_t SimpleRecurrentLayer::getParamCount() {
	return 5;
}

void SimpleRecurrentLayer::loadParams(Tensor params[]) {
	weightsInput = params[0];
	weightsHiddenPast = params[1];
	weightsHiddenPresent = params[2];
	biasesHidden = params[3];
	biasesOutput = params[4];
}

void SimpleRecurrentLayer::fetchParams(Tensor params[]) {
	params[0] = weightsInput;
	params[1] = weightsHiddenPast;
	params[2] = weightsHiddenPresent;
	params[3] = biasesHidden;
	params[4] = biasesOutput;
}

void SimpleRecurrentLayer::getParamsSizes(Size sizes[]) {
	sizes[0] = Size((size_t)2, inputSize, size);
	sizes[1] = Size((size_t)2, 1, size);
	sizes[2] = Size((size_t)2, 1, size);
	sizes[3] = Size((size_t)2, 1, size);
	sizes[4] = Size((size_t)2, 1, size);
}

size_t SimpleRecurrentLayer::getStateCount() {
	return 1;
}

void SimpleRecurrentLayer::loadState(Tensor state[]) {
	hiddenLayer = state[0];
}

void SimpleRecurrentLayer::fetchStates(Tensor states[]) {
	states[0] = hiddenLayer;
}

void SimpleRecurrentLayer::getStateSizes(Size sizes[]) {
	sizes[0] = Size((size_t)2, (size_t)1, size);
}

Tensor SimpleRecurrentLayer::getValue() {
	return layer.slice(0,lastSize);
}
