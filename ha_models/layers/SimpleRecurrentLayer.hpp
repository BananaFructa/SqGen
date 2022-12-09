#pragma once

#include "Layer.hpp"

class SimpleRecurrentLayer : public Layer {
private:

	size_t lastSize = 0;

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

	Size getInputSize();
	Size getOutputSize();
	size_t getParamCount();
	size_t getStateCount();

	unsigned short stepCount();
	unsigned short stepAsync(Tensor input);

	void rndParams(CurandManager& curandManager);
	void loadParams(Tensor params[]);
	void loadState(Tensor state[]);
	void getParamsSizes(Size sizes[]);
	void getStateSizes(Size sizes[]);
	void fetchParams(Tensor params[]);
	virtual void fetchStates(Tensor states[]);

	Tensor getValue();

};

