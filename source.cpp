#include <iostream>

#include "sqgen/Simulation.hpp"

int main() {

	Simulation simulation;

	simulation.addNewAgent();

	simulation.AP_Manager.compile(&simulation.agents[0], 1);

	Tensor input(Size(3, 1, 10, 1));

	float a[10] = { 1,-1,1,0,0,0,0,0,0,0 };

	float b[8];

	input.setValue(a);

	simulation.AP_Manager.predict(input).getValue(b);

	for (int i = 0; i < 8; i++) std::cout << b[i] << ' ';

	return 0;
}