#pragma once

#include <vector>
#include <map>

#include "Random.hpp"
#include "RndPosGenerator.h"
#include "Constant.h"
#include "Agent.hpp"
#include "Position.h"
#include "NNAgentModelManager.hpp"
#include "../ha_models/ReferenceMappedTensor.hpp"
#include "../ha_models/TensorMemAllocator.hpp"

/*

	INPUT LAYER:

	=
	|
	| 1x Current food value
	|
	=
	|
	| 1x Current food tile value
	|
	=
	|
	| 4x Visual directional latent space
	|
	=
	|
	| 4x Signal directional average
	|
	=

	OUTPUT LAYER:

	=
	|
	| 1x Eat decision
	|
	=
	|
	| 1x Multiply decision
	|
	=
	|
	| 4x UP/DOWN/RIGHT/LEFT
	|
	=
	|
	| 1x Attack decision
	|
	=
	|
	| 1x Share decision
	|
	=
	|
	| 1x Change signal
	|
	=

*/

struct Simulation {
private:
public:
//  =======================================================================

	RndPosGenerator randomPositionGenerator;

//  =======================================================================

	CurandManager curandManager = CurandManager(Constants::curandPoolSize, Constants::seed);

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

//  =======================================================================

public:

	std::vector<Agent> agents;

	Simulation();

	void addNewAgent();
	void addAgent(Agent parent);
	void removeAgent(size_t index);

	void update();

	bool positionOccupied(Position pos);

	AgentID getAgentID();
	SpecieID getSpecieID();

};