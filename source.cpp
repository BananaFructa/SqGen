#include <iostream>

#include "./ha_models/NNModel.hpp"
#include "./ha_models/layers/DenseLayer.hpp"
#include "./ha_models/random/CurandManager.h"
#include "./ha_models/Size.h"


// TODO: I dont think layers have to passed as pointers

int main() {


	TENSOR_TYPE a1[18] = {
		1,0,1,
		0,1,0,
		1,0,1,

		1,0,0,
		0,1,0,
		0,0,1,
	};

	TENSOR_TYPE a2[9] = {
		1,2,3,
		4,5,6,
		7,8,9
	};

	TENSOR_TYPE a3[1] = { 2 };

	TENSOR_TYPE res[18] = { 0 };

	Tensor input(Size(2, 1, 1));
	input.setValue(a3);

	CurandManager curand(20,100);

	gpuSync();

	NNModel model(1);

	model.addLayer(DenseLayer(1, 5, Func::SIGMOID));
	model.addLayer(DenseLayer(5, 10, Func::SIGMOID));
	model.addLayer(DenseLayer(10, 1, Func::SIGMOID));

	model.randomizeUniform(curand);

	gpuSync();

	model.predict(input);

	gpuSync();

	model.getPrediction().getValue(res);

	for (int i = 0; i < 18; i++) std::cout << res[i] << " ";

	getchar();

	return 0;

}