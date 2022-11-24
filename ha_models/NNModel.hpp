#pragma once

#include <vector>
#include "layers/Layer.hpp"

struct NNModel {

	std::vector<Layer*> layers;

public:

	size_t poolSize = 0;


	NNModel(size_t poolSize);
	~NNModel();

	void addLayer(Layer* layer);
	bool takeAsyncStep(TensorPool2D& input, size_t currentLayer);
	void predict(TensorPool2D& input);
	void free();
	void freeLayers();
	size_t layerCount();
	TensorPool2D& getPrediction();

	void randomize();


};