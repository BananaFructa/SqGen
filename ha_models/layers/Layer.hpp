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

	virtual size_t getInputSize() { return 0; };
	virtual size_t getOutputSize() { return 0; };

	virtual void rndParams(CurandManager& curandManager) {};
	virtual size_t loadParams(Tensor params[]) { return 0; };
	virtual size_t loadState(Tensor states[]) { return 0; };

	virtual unsigned short stepCount() { return 0; };
	virtual unsigned short stepAsync(Tensor& input) { return 0; };

	virtual Tensor& getValue() { return Tensor::EmptyTensor; };

};