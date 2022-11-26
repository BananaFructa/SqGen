#pragma once

#include "../Tensor.hpp"
#include "../random/CurandManager.h"

class Layer {
public:

	virtual void free() {};
	virtual void freeLayers() {};

	virtual void setPool(size_t newSize) {};

	virtual size_t getInputSize() { return 0; };
	virtual size_t getOutputSize() { return 0; };

	virtual void rndParams(CurandManager& curandManager) {};
	virtual void loadParams(Tensor params[]) {};

	virtual unsigned short stepCount() { return 0; };
	virtual unsigned short stepAsync(Tensor& input) { return 0; };

	virtual Tensor& getValue() { return Tensor::EmptyTensor; };

};

