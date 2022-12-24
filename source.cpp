#include <iostream>

#include "sqgen/Simulation.hpp"
#include "sqgen/graphics/RenderManager.hpp"

// 16.11.2022

// TODO: graphical interface

int main() {


	Simulation simulation;

	//simulation.setAgentPos(0, Position(3, 7));

	RenderManager renderMananger(simulation);

	for (int i = 0; i < Constants::startingAgentCount; i++) {
		if ((i + 1) % 1000 == 0) std::cout << "Generating agents " << (i + 1) << '\n';
		simulation.addNewAgent();
	}

	std::thread([&]() {
		for (;;) {
			simulation.update();
			if (!simulation.paused) simulation.printProfilerInfo();
			renderMananger.updateRenderData();
		}
	}).detach();

	renderMananger.RenderLoop();

	return 0;
}