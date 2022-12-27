#include <iostream>

#include "sqgen/Simulation.hpp"
#include "sqgen/graphics/RenderManager.hpp"
#include "sqgen/Rational.hpp"

// 16.11.2022

// TODO: delta signal view

int main() {

	Simulation simulation;

	//simulation.setAgentPos(0, Position(3, 7));

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
			if (!simulation.paused) simulation.printProfilerInfo();
			renderMananger.updateRenderData();
		}
	}).detach();

	renderMananger.RenderLoop();

	return 0;
}