#include "NNModel.hpp"

NNModel::NNModel(size_t poolSize) {
	this->poolSize = poolSize;
}

NNModel::~NNModel() {
	for (size_t i = 0; i < layers.size(); i++) delete layers[i];
}

void NNModel::addLayer(Layer* layer) {
	layer->setPool(poolSize);
	layers.push_back(layer);
}

bool NNModel::takeAsyncStep(TensorPool2D& input, size_t currentLayer) {
	return layers[currentLayer]->stepAsync(input) == 0;
}

void NNModel::predict(TensorPool2D& input) {
	for (size_t i = 0; i < layerCount(); i++) {
		while (!takeAsyncStep(input, i)) gpuSync();
	}
}

void NNModel::free() {
	for (int i = 0; i < layers.size(); i++) layers[i]->free();
}

void NNModel::freeLayers() {
	for (int i = 0; i < layers.size(); i++) layers[i]->freeLayers();
}

size_t NNModel::layerCount() {
	return layers.size();
}

TensorPool2D& NNModel::getPrediction() {
	return layers[layerCount() - 1]->getValue();
}

void NNModel::randomize() {
	for (int i = 0; i < layers.size(); i++) layers[i]->rndParams();
}
