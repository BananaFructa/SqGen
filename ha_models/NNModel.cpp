#include "NNModel.hpp"

NNModel::NNModel(size_t poolSize) {
	this->poolSize = poolSize;
}

NNModel::~NNModel() {
	for (int i = 0; i < layers.size(); i++) layers[i]->free();
}

void NNModel::addLayer(Layer* layer) {
	layer->setPool(poolSize);
	layers.push_back(layer);
}

bool NNModel::takeAsyncStep(Tensor& input, size_t currentLayer) {
	if (currentLayer == 0) {
		return layers[currentLayer]->stepAsync(input) == 0;
	}
	else {
		Tensor sliced = layers[currentLayer - 1]->getValue().slice(0, input.size.getDimSize(input.size.dim - 1));
		return layers[currentLayer]->stepAsync(sliced) == 0;
	}
}

void NNModel::predict(Tensor& input) {
	for (size_t i = 0; i < layerCount(); i++) {
		while (!takeAsyncStep(input, i)) gpuSync();
	}
}

void NNModel::free() {
	for (int i = 0; i < layers.size(); i++) layers[i]->free();
}

size_t NNModel::layerCount() {
	return layers.size();
}

Tensor& NNModel::getPrediction() {
	return layers[layerCount() - 1]->getValue();
}

void NNModel::randomizeUniform(CurandManager& curandManager) {
	for (int i = 0; i < layerCount(); i++) layers[i]->rndParams(curandManager);
}

void NNModel::loadModel(Tensor variables[]) {
	size_t current = 0;
	for (int i = 0; i < layerCount(); i++) {
		current += layers[i]->loadParams(&variables[current]);
	}
}

void NNModel::disableDefInternalAlloc() {
	for (int i = 0; i < layerCount(); i++) layers[i]->disableDefInternalAlloc();
}
