#pragma once

#include "../Tensor2D.hpp"
#include "../TensorPool2D.hpp"

class Layer {
public:

	virtual void free() {};
	virtual void freeLayers() {};

	virtual void setPool(size_t newSize) {};

	virtual size_t getInputSize() { return 0; };
	virtual size_t getOutputSize() { return 0; };

	virtual void rndParams() {};
	virtual void loadParams(Tensor2D params[]) {};

	virtual unsigned short stepCount() { return 0; };
	virtual unsigned short stepAsync(TensorPool2D& input) { return 0; };

	virtual TensorPool2D& getValue() { return TensorPool2D::EmptyPool; };

};

