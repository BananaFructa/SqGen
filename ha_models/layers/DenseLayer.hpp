#pragma once

#include "Layer.hpp"
#include "../random/CurandManager.h"

class DenseLayer : public Layer {
private:

	Tensor layer;
	Tensor weights;
	Tensor biases;

	// Used for running softmax
	Tensor auxSumMem;

public:

	size_t size = 0;
	size_t inputSize = 0;
	unsigned short step = 0;
	Activation activation;

	DenseLayer(size_t getInputSize, size_t size, Activation activation);

	void free();

	void setPool(size_t newSize);

	size_t getInputSize();
	size_t getOutputSize();

	unsigned short stepCount();
	unsigned short stepAsync(Tensor& input);

	void rndParams(CurandManager& curandManager);
	size_t loadParams(Tensor params[]);

	Tensor& getValue();

};