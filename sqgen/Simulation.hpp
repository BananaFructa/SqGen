#pragma once

#include <vector>

#include <map>
#include <cmath>
#include "Constant.h"
#include "Agent.hpp"
#include "Position.h"
#include "NNAgentModelManager.hpp"
#include "../ha_models/ReferenceMappedTensor.hpp"
#include "../ha_models/TensorMemAllocator.hpp"


struct Simulation {
private:
//  =======================================================================

	CurandManager curandManager = CurandManager(Constants::curandPoolSize, Constants::curandSeed);

	// Specie Information Encoder
	NNModel SIE_Network = NNModel(Constants::nnPoolSize);
	NNAgentModelManager SIE_Manager;

	// Action Processing
	NNModel AP_Netowrk = NNModel(Constants::nnPoolSize);
	NNAgentModelManager AP_Manager;

	void buildSIE(NNModel& model);
	void buildAP(NNModel& model);

//  =======================================================================

	Tensor gpuFoodMap = Tensor(Size(2, Constants::mapSize, Constants::mapSize));
	Tensor gpuSignalMap = Tensor(Size(2, Constants::mapSize, Constants::mapSize));
	ReferenceMappedTensor gpuSpecieSignalMap = ReferenceMappedTensor(Size(2, Constants::spicieSignalCount, Constants::totalMapSize));

	std::map<SpecieID, Tensor> specieSignalDict;

	TensorMemAllocator<Tensor> specieSignalAllocator = TensorMemAllocator<Tensor>(Size(Size(1, Constants::spicieSignalCount)));

//  =======================================================================

	SpecieID specieCounter = 0;
	AgentID lastAgentID = 0;
	std::vector<AgentID> avalabileAgentIDs;
	std::map<SpecieID, size_t> specieInstaceCounter;

public:

	std::vector<Agent> agents;

	Simulation();

	void addNewAgent();
	void addAgent(Agent parent);
	void removeAgent();

	void createNewSpecie();
	void createSpecie(SpecieID parentSpecie);
	void removeSpecie();

	AgentID getAgentID();
	SpecieID getSpecieID();

};