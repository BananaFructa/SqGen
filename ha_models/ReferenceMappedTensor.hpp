#pragma once

#include "Tensor.hpp"

class ReferenceMappedTensor : public Tensor {
public:
	ReferenceMappedTensor();
	ReferenceMappedTensor(Size size);

	void init(Size size);
	void setRef(size_t index, Tensor& t);
	void setRef(size_t index, Scalar s);
	void swap(size_t a, size_t b);

};

