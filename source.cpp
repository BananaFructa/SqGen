#include <iostream>

#include "./ha_models/NNModel.hpp"
#include "./ha_models/layers/DenseLayer.hpp"
#include "./ha_models/random/CurandManager.h"
#include "./ha_models/Size.h"

int main() {

	TENSOR_TYPE a3[1] = { 2 };

	TENSOR_TYPE res[18] = { 0 };

	Tensor input(Size(3, 1, 1, 1));
	input.setValue(a3);

	CurandManager curand(20, 100);

	gpuSync();

	NNModel model(100);

	model.addLayer(new DenseLayer(1, 5, Activation::SIGMOID));
	model.addLayer(new DenseLayer(5, 10, Activation::SIGMOID));
	model.addLayer(new DenseLayer(10, 1, Activation::SOFTMAX));

	model.randomizeUniform(curand);

	gpuSync();

	model.predict(input);

	gpuSync();

	model.getPrediction().slice(0,18).getValue(res);

	for (int i = 0; i < 18; i++) std::cout << res[i] << " ";

	getchar();

	return 0;

}