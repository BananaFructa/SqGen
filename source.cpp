#include <iostream>

#include "sqgen/Simulation.hpp"
#include "sqgen/graphics/RenderManager.hpp"

// 16.11.2022

// TODO: graphical interface

int main() {


	Simulation simulation;

	for (int i = 0; i < 20;i++) simulation.addNewAgent();

	//simulation.setAgentPos(0, Position(3, 7));

	RenderManager renderMananger(simulation);

	std::thread([&]() {
		for (;;) {
			simulation.update();
			simulation.printProfilerInfo();
			renderMananger.updateRenderData();
		}
	}).detach();

	renderMananger.RenderLoop();

	return 0;
}