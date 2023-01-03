#include <iostream>

#include "sqgen/Simulation.hpp"
#include "sqgen/graphics/RenderManager.hpp"
#include "sqgen/Rational.hpp"

#include "ha_models/layers/DenseLayer.hpp"

// 16.11.2022

// TODO: delta signal view

int main() {

	//return 0; //

	char path[] = "D:/Repos/SqGen/SqGen/codependence_train/B_SG_NPY";

	NNModel model(1);

	model.addLayer(new DenseLayer(10, 10, Activation::TANH));
	model.addLayer(new DenseLayer(10, 10, Activation::TANH));
	model.addLayer(new DenseLayer(10, 1, Activation::TANH));

	model.loadModel(path);

	float fin[10] = { 0.4,1,	  0,0,0.5,0,		0,0,0,0 };

	Tensor input(Size(3, 1, 10, 1));

	input.setValue(fin);

	float out[1];

	model.predict(input);
	gpuSync();
	model.getPrediction().getValue(out);
	for (int i = 0;i < 1;i++)
	std::cout << out[i] << " ";

	return 0;

	Simulation simulation;

	simulation.loadSpecie("codependence_build");
	simulation.addAgentFromSpecie(1, Position(0, 0));

	//simulation.setAgentPos(0, Position(3, 7));

	RenderManager renderMananger(simulation);

	std::thread([&]() {
		for (;;) {
			if (simulation.agents.size() == 0) {
				simulation.restartFoodMap();
				//for (int i = 0; i < Constants::startingAgentCount; i++) {
				//	if ((i + 1) % 1000 == 0) std::cout << "Generating agents " << (i + 1) << '\n';
				//	simulation.addNewAgent();
				//}
			}
			simulation.update();
			if (!simulation.paused) simulation.printProfilerInfo();
			renderMananger.updateRenderData();
		}
	}).detach();

	renderMananger.RenderLoop();

	return 0;
}