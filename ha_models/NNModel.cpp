#include "NNModel.hpp"

#include <string>

NNModel::NNModel(){
}

NNModel::NNModel(size_t poolSize) {
	this->poolSize = poolSize;
}

void NNModel::addLayer(Layer* layer) {
	if (!internalAlloc) layer->disableDefInternalAlloc();
	layer->setPool(poolSize);
	layers.push_back(layer);
	paramCount += layer->getParamCount();
	stateCount += layer->getStateCount();
}

bool NNModel::takeAsyncStep(Tensor& input, size_t currentLayer) {
	if (currentLayer == 0) {
		return layers[currentLayer]->stepAsync(input) == 0;
	}
	else {
		return layers[currentLayer]->stepAsync(layers[currentLayer - 1]->getValue()) == 0;
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

Tensor NNModel::getPrediction() {
	return layers[layerCount() - 1]->getValue();
}

void NNModel::randomizeUniform(CurandManager& curandManager) {
	for (int i = 0; i < layerCount(); i++) layers[i]->rndParams(curandManager);
}

void NNModel::setModelParams(Tensor variables[]) {
	size_t current = 0;
	for (int i = 0; i < layerCount(); i++) {
		if (layers[i]->getParamCount() == 0) continue;
		layers[i]->loadParams(&variables[current]);
		current += layers[i]->getParamCount();
	}
}

void NNModel::setModelStates(Tensor states[]) {
	size_t current = 0;
	for (int i = 0; i < layerCount(); i++) {
		if (layers[i]->getStateCount() == 0) continue;
		layers[i]->loadState(&states[current]);
		current += layers[i]->getStateCount();
	}
}

void NNModel::getModelParams(Tensor variables[]) {
	size_t current = 0;
	for (int i = 0; i < layerCount(); i++) {
		if (layers[i]->getParamCount() == 0) continue;
		layers[i]->fetchParams(&variables[current]);
		current += layers[i]->getParamCount();
	}
}

void NNModel::getModelStates(Tensor states[]) {
	size_t current = 0;
	for (int i = 0; i < layerCount(); i++) {
		if (layers[i]->getStateCount() == 0) continue;
		layers[i]->fetchStates(&states[current]);
		current += layers[i]->getStateCount();
	}
}

void NNModel::loadModel(const char* path) {

	std::string spath(path);

	std::vector<Tensor> params(paramCount);

	for (size_t i = 0; i < paramCount; i++) {
		params[i].load((spath + "/" + std::to_string(i) + ".npy").c_str());
	}

	setModelParams(params.data());

}

void NNModel::loadState(const char* path) {
	std::string spath(path);

	std::vector<Tensor> states(paramCount);

	for (size_t i = 0; i < paramCount; i++) {
		states[i].load((spath + "/" + std::to_string(i) + ".npy").c_str());
	}

	setModelStates(states.data());
}

void NNModel::disableDefInternalAlloc() {
	internalAlloc = false;
}

std::vector<Layer*>& NNModel::getLayers() {
	return layers;
}
