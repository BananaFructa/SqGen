#pragma once

#include <vector>
#include "layers/Layer.hpp"

struct NNModel {

	std::vector<Layer*> layers;
	bool internalAlloc = true;

public:

	size_t poolSize = 0;

	size_t paramCount = 0;
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

	void setModelParams(Tensor variables[]);
	void setModelStates(Tensor states[]);
	void getModelParams(Tensor variables[]);
	void getModelStates(Tensor states[]);
	void loadModel(const char* path);
	void loadState(const char* path);

	void disableDefInternalAlloc();

	std::vector<Layer*>& getLayers();


};