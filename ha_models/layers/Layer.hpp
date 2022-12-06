#pragma once

#include "../Tensor.hpp"
#include "../random/CurandManager.h"

enum Activation {
	ReLU,
	SIGMOID,
	TANH,
	SOFTMAX
};

class Layer {
protected:

	bool allocInternal = true;

public:

	virtual void free() {};

	void disableDefInternalAlloc() {
		allocInternal = false;
	}

	virtual void setPool(size_t newSize) {};

	virtual Size getInputSize() { return Size(0); };
	virtual Size getOutputSize() { return Size(0); };
	virtual size_t getParamCount() { return 0; };
	virtual size_t getStateCount() { return 0; };

	virtual void rndParams(CurandManager& curandManager) {};
	virtual void getParamsSizes(Size sizes[]) {};
	virtual void getStateSizes(Size sizes[]) {};
	virtual void loadParams(Tensor params[]) {};
	virtual void loadState(Tensor states[]) {};
	virtual void fetchParams(Tensor params[]) {};
	virtual void fetchStates(Tensor states[]) {};

	virtual unsigned short stepCount() { return 0; };
	virtual unsigned short stepAsync(Tensor input) { return 0; };

	virtual Tensor getValue() { return Tensor::EmptyTensor; };

};