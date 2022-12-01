#include <iostream>

#include "./ha_models/NNModel.hpp"
#include "./ha_models/layers/DenseLayer.hpp"
#include "./ha_models/random/CurandManager.h"
#include "./ha_models/Size.h"`
#include "./ha_models/UniquelyMappedTensor.hpp"
#include "./ha_models/ReferenceMappedTensor.hpp"

int main() {

	TENSOR_TYPE a[] = { 
				1,1,1,
				1,1,1,
				1,1,1,
	};

	TENSOR_TYPE b[] = {
		1,0,0,
		0,1,0,
		0,0,1,
		1,1,1,
		1,1,1,
		1,1,1
	};

	TENSOR_TYPE c[]{
		2,2,2,
		2,2,2,
		2,2,2
	};

	TENSOR_TYPE res[18] = { 0 };

	ReferenceMappedTensor t(Size(3, 3, 3, 2));
	Tensor t1(Size(2, 3, 3));
	Tensor t2(Size(2, 3, 3));
	Tensor t3(Size(3, 3, 3, 2));

	t1.setValue(a);
	t2.setValue(c);
	t3.setValue(b);

	t.setRef(0, t2);
	t.setRef(1, t1);
	t.syncMap();

	(Tensor)t = t.slice(1,2) % t3;

	t.getValue(res);


	gpuSync();

	for (int i = 0; i < 18; i++) {
		std::cout << res[i] << " ";
	}

	getchar();

	return 0;

}