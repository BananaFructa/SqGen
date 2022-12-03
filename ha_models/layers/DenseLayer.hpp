#pragma once

#include "Layer.hpp"
#include "../random/CurandManager.h"

class DenseLayer : public Layer {
private:

	size_t lastSize;

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

	Size getInputSize();
	Size getOutputSize();
	size_t getParamCount();

	unsigned short stepCount();
	unsigned short stepAsync(Tensor input);

	void rndParams(CurandManager& curandManager);
	void loadParams(Tensor params[]);
	void getParamsSizes(Size sizes[]);

	Tensor getValue();

};