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

	simulation.update();

	gpuSync();

	float sig[12];

	Tensor sliced = simulation.SIE_InputPool.slice(0, 12);
	simulation.SIE_Manager.predict(sliced).getValue(sig);

	gpuSync();

	for (int i = 0; i < 12; i++) std::cout << sig[i] << " ";

	return 0;
}