#pragma once

#include "Layer.hpp"

#include "../TensorPool2D.hpp"

#define DENSE_LAYER_STEP_COUNT 3

class DenseLayer : public Layer {
private:

	TensorPool2D layer;
	Tensor2D weights;
	Tensor2D biases;

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
	unsigned short stepAsync(TensorPool2D& input);

	void rndParams();
	void loadParams(Tensor2D params[]);

	TensorPool2D& getValue();

};