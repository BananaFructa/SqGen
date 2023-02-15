#include <iostream>

#include "sqgen/Simulation.hpp"
#include "sqgen/graphics/RenderManager.hpp"
#include "sqgen/Rational.hpp"

#include "ha_models/layers/DenseLayer.hpp"

#include <highfive/H5File.hpp>
#include <fstream>

// 16.11.2022

int ticks = 1;

// TODO: SAVE COLORS
int main() {

	/*Tensor input;
	input.load("input.npy");
	input = input.slice(1831,1832);
	Tensor params;
	params.load("params.npy");
	Tensor biases;
	biases.load("biases.npy");

	Tensor result(Size(1, 10, 15851));

	result = input * params;
	gpuSync();
	result = result + biases;
	gpuSync();
	result.functionPass(KERNEL_TANH);
	gpuSync();

	std::vector<float> values(result.size.size);
	result.getValue(values.data());

	return 0;*/

	Simulation simulation;

	//simulation.loadSpecie("codependence_build/A");
	//simulation.loadSpecie("codependence_build/B");
	//for (int j = 0;j < 20;j++)
	//for (int i = 0; i < 20; i++) {
	//	simulation.addAgentFromSpecie(1, Position(i * 20, j * 20));
	//	simulation.addAgentFromSpecie(2, Position(i*20, j * 20 + 2));
	//}
	RenderManager renderMananger(simulation);

	std::thread([&]() {
		for (;;) {
			if (simulation.agents.size() == 0) {
				if ((std::ifstream("simulationState.h5")).good()) {
					simulation.loadSimulationState("simulationState.h5");
				}
				else {
					simulation.restartFoodMap();
					for (int i = 0; i < Constants::startingAgentCount; i++) {
						if ((i + 1) % 1000 == 0) std::cout << "Generating agents " << (i + 1) << '\n';
						simulation.addNewAgent();
					}
				}
			}
			if (renderMananger.shouldSave || (!simulation.paused && (ticks++) % 5000 == 0)) {
				simulation.saveSimulationState("simulationState.h5");
				renderMananger.shouldSave = false;
			}
			simulation.update();
			renderMananger.updateRenderData();
		}
	}).detach();

	renderMananger.RenderLoop();

	return 0;
}