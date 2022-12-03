#pragma once

#include <vector>
#include "layers/Layer.hpp"

struct NNModel {

	std::vector<Layer*> layers;

public:

	size_t poolSize = 0;

	size_t variableCount = 0;
	size_t stateCount = 0;

	NNModel();
	NNModel(size_t poolSize);

	void addLayer(Layer* layer);
	bool takeAsyncStep(Tensor& input, size_t currentLayer);
	void predict(Tensor& input);
	void free();
	size_t layerCount();
	Tensor getPrediction();

	void randomizeUniform(CurandManager& curandManager);
	void loadModel(Tensor variables[]);
	void loadState(Tensor states[]);

	void disableDefInternalAlloc();

	std::vector<Layer*>& getLayers();


};