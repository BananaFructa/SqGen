#include <iostream>

#include "sqgen/Simulation.hpp"
#include "sqgen/graphics/RenderManager.hpp"

// 16.11.2022

// TODO: graphical interface

int main() {


	Simulation simulation;

	RenderManager renderMananger(simulation);

	renderMananger.RenderLoop();


	return 0;
}