#include <iostream>

#include "./ha_models/NNModel.hpp"
#include "./ha_models/layers/DenseLayer.hpp"
#include "./ha_models/random/CurandManager.h"
#include "./ha_models/Size.h"`
#include "./ha_models/UniquelyMappedTensor.hpp"

int main() {

	TENSOR_TYPE a3[] = { 
				1,1,1,
				1,1,1,
				1,1,1,
				2,2,2,
				2,2,2,
				2,2,2
	};

	TENSOR_TYPE res[18] = { 0 };

	UniquelyMappedTensor t(Size(3, 3, 3, 2));

	t.setValue(a3);

	t.swap(0, 1);

	t.syncMap();

	t.slice(0,1).getValue(res);

	gpuSync();

	for (int i = 0; i < 18; i++) {
		std::cout << res[i] << " ";
	}

	getchar();

	return 0;

}