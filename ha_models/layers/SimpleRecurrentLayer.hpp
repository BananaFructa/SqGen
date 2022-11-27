#pragma once

#include "Layer.hpp"

class SimpleRecurrentLayer : public Layer {
private:

	Tensor weightsInput;

	Tensor weightsHiddenPast;
	Tensor weightsHiddenPresent;

	Tensor biasesHidden;
	Tensor biasesOutput;

	Tensor hiddenLayer;
	Tensor layer;

public:

	size_t size = 0;
	size_t inputSize = 0;
	unsigned short step = 0;

	Activation activation;
	Activation hiddenActivation;

	SimpleRecurrentLayer(size_t inputSize, size_t size, Activation activation, Activation hiddenActivation);

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

