#pragma once

#include <vector>
#include "layers/Layer.hpp"

struct NNModel {

	std::vector<Layer*> layers;

public:

	~NNModel();

	void addLayer(Layer* layer);
	bool takeAsyncStep(TensorPool2D& input, size_t currentLayer);
	void predict(TensorPool2D& input);
	size_t layerCount();
	TensorPool2D& getPrediction();

	Layer* swapLayers(Layer* layer, size_t index);


};