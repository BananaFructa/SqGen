#pragma once

#include "Layer.hpp"
#include "../random/CurandManager.h"

#define DENSE_LAYER_STEP_COUNT 3

class DenseLayer : public Layer {
private:

	Tensor layer;
	Tensor weights;
	Tensor biases;

public:

	size_t size = 0;
	size_t inputSize = 0;
	unsigned short step = 0;
	Func activation;

	DenseLayer(size_t getInputSize, size_t size, Func activation);

	void free();
	void freeLayers();

	void setPool(size_t newSize);

	size_t getInputSize();
	size_t getOutputSize();

	unsigned short stepCount();
	unsigned short stepAsync(Tensor& input);

	void rndParams(CurandManager& curandManager);
	void loadParams(Tensor params[]);

	Tensor& getValue();

};