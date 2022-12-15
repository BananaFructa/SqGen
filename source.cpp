#include <iostream>

#include "sqgen/Simulation.hpp"

// 16.11.2022

int main() {

	Simulation simulation;

	simulation.addNewAgent();
	simulation.addNewAgent();
	simulation.addNewAgent();

	simulation.setAgentPos(0, Position(5, 5));
	simulation.setAgentPos(1, Position(5, 4));
	simulation.setAgentPos(2, Position(4, 4));

	simulation.signalMap[4 + 6 * Constants::mapSize] = 1;

	simulation.update();

	return 0;
}