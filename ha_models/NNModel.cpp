#include "NNModel.hpp"

NNModel::~NNModel() {
	for (size_t i = 0; i < layers.size(); i++) delete layers[i];
}

void NNModel::addLayer(Layer* layer) {
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

size_t NNModel::layerCount() {
	return layers.size();
}

TensorPool2D& NNModel::getPrediction() {
	return layers[layerCount() - 1]->getValue();
}

Layer* NNModel::swapLayers(Layer* layer, size_t index) {
	Layer* old = layers[index];
	layers[index] = layer;
	return old;
}
