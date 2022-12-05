#include <iostream>

#include "sqgen/NNAgentModelManager.hpp"

#include "ha_models/layers/DenseLayer.hpp"
#include "ha_models/layers/SimpleRecurrentLayer.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int main() {

	CurandManager curandManager(100, 105);

	NNModel model(100'000);

	model.disableDefInternalAlloc();

	model.addLayer(new DenseLayer(1, 10, Activation::ReLU));
	model.addLayer(new DenseLayer(10, 100, Activation::ReLU));
	model.addLayer(new SimpleRecurrentLayer(100, 100, Activation::TANH,Activation::TANH));
	model.addLayer(new DenseLayer(100, 10, Activation::ReLU));
	model.addLayer(new DenseLayer(10, 1, Activation::ReLU));

	NNAgentModelManager modelManager(model, curandManager);

	modelManager.registerNewSpiece(0, -1, 1);
	modelManager.registerNewSpiece(1, -1, 1);
	modelManager.registerSpecie(0, 2, 0.2,-1,1);

	modelManager.registerAgent(5);
	modelManager.registerAgent(1);
	modelManager.registerAgent(2);

	Agent agents[] = { {1,5,0,0},{1,2,0,0} ,{1,1,0,0} };

	modelManager.compile(agents, 3);

	Tensor input(Size(3, 1, 1, 3));

	TENSOR_TYPE a[3] = {
		0,0,0
	};

	TENSOR_TYPE b[3] = { 1,1,1 };

	for (;;) {

		input.setValue(a);

		gpuSync();

		Tensor out = modelManager.predict(input);

		gpuSync();

		out.getValue(b);

		std::cout << b[0] << " " << b[1] << " " << b[2] << '\n';
	}

		gpuErrchk(cudaPeekAtLastError());

	return 0;
}