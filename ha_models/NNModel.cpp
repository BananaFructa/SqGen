#include "NNModel.hpp"

NNModel::NNModel(size_t poolSize) {
	this->poolSize = poolSize;
}

void NNModel::addLayer(Layer layer) {
	layer.setPool(poolSize);
	layers.push_back(layer);
}

bool NNModel::takeAsyncStep(Tensor& input, size_t currentLayer) {
	return layers[currentLayer].stepAsync(currentLayer == 0 ? input : layers[currentLayer - 1].getValue()) == 0;
}

void NNModel::predict(Tensor& input) {
	for (size_t i = 0; i < layerCount(); i++) {
		while (!takeAsyncStep(input, i)) gpuSync();
	}
}

void NNModel::free() {
	for (int i = 0; i < layers.size(); i++) layers[i].free();
}

void NNModel::freeLayers() {
	for (int i = 0; i < layers.size(); i++) layers[i].freeLayers();
}

size_t NNModel::layerCount() {
	return layers.size();
}

Tensor& NNModel::getPrediction() {
	return layers[layerCount() - 1].getValue();
}

void NNModel::randomizeUniform(CurandManager& curandManager) {
	for (int i = 0; i < layers.size(); i++) layers[i].rndParams(curandManager);
}
