#include <iostream>

#include "sqgen/Simulation.hpp"
#include "sqgen/graphics/RenderManager.hpp"
#include "sqgen/Rational.hpp"

#include "ha_models/layers/DenseLayer.hpp"

// 16.11.2022

// TODO: fix the food kernel compile

int main() {

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
				simulation.restartFoodMap();
				for (int i = 0; i < Constants::startingAgentCount; i++) {
					if ((i + 1) % 1000 == 0) std::cout << "Generating agents " << (i + 1) << '\n';
					simulation.addNewAgent();
				}
			}
			simulation.update();
			renderMananger.updateRenderData();
		}
	}).detach();

	renderMananger.RenderLoop();

	return 0;
}