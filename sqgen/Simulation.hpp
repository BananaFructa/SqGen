#pragma once

#include <vector>

#include "Agent.hpp"
#include "NNAgentModelManager.hpp"


struct Simulation {
private:
//  =======================================================================

	const size_t nnPoolSize = 100'000;

	const size_t curandPoolSize = 1000;
	const unsigned long curandSeed = 123;

	CurandManager curandManager = CurandManager(curandPoolSize, curandSeed);

	// Specie Information Encoder
	NNModel SIE_Network = NNModel(nnPoolSize);
	NNAgentModelManager SIE_Manager;

	// Action Processing
	NNModel AP_Netowrk = NNModel(nnPoolSize);
	NNAgentModelManager AP_Network;

	void buildSIE(NNModel& model);
	void buildAP(NNModel& model);

//  =======================================================================

	const size_t mapSize = 50;

	Tensor gpuMapSet = Tensor(Size(3, mapSize, mapSize, 3));


	// TODO: figure out signal id gpu process

public:

	std::vector<Agent> agents;

	Simulation();
	void removeAgent();

};