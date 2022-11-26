#pragma once

#include <vector>
#include "layers/Layer.hpp"

struct NNModel {

	std::vector<Layer> layers;

public:

	size_t poolSize = 0;


	NNModel(size_t poolSize);

	void addLayer(Layer layer);
	bool takeAsyncStep(Tensor& input, size_t currentLayer);
	void predict(Tensor& input);
	void free();
	void freeLayers();
	size_t layerCount();
	Tensor& getPrediction();

	void randomizeUniform(CurandManager& curandManager);


};