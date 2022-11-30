#pragma once

#include "Tensor.hpp"

class UniquelyMappedTensor : public Tensor {

public:

	UniquelyMappedTensor();
	UniquelyMappedTensor(Size size);

	void init(Size size);

};

