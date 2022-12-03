#include <iostream>

#include "sqgen/NNAgentModelManager.hpp"

#include "ha_models/layers/DenseLayer.hpp"

int main() {

	CurandManager curandManager(100'000, 100);

	NNModel model(100'000);

	model.addLayer(new DenseLayer(1, 100, Activation::ReLU));
	model.addLayer(new DenseLayer(100, 100, Activation::ReLU));
	model.addLayer(new DenseLayer(100, 1, Activation::ReLU));

	NNAgentModelManager modelManager(model, curandManager);

	modelManager.registerNewSpiece(0, -1, 1);
	modelManager.registerNewSpiece(1, -1, 1);
	modelManager.registerSpecie(0, 2, 0.01, -10, 1);

	Agent agents[] = { {0,0,0,0},{1,0,0,0},{2,0,0,0} };

	modelManager.compile(agents, 3);

	Tensor input(Size(3, 1, 1, 3));

	TENSOR_TYPE a[3] = {
		1,1,1
	};

	input.setValue(a);

	gpuSync();

	Tensor out = modelManager.predict(input);

	out.getValue(a);

	std::cout << a[0] << " " << a[1] << " " << a[2];

	return 0;
}