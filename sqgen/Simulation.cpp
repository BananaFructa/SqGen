#include "Simulation.hpp"

#include "../ha_models/layers/DenseLayer.hpp"
#include "../ha_models/layers/SimpleRecurrentLayer.hpp"

void Simulation::buildSIE(NNModel& model) {
	// Simple test arhitecture
	model.addLayer(new DenseLayer(Constants::spicieSignalCount, 15, Activation::TANH));
	model.addLayer(new DenseLayer(15, 15, Activation::TANH));
	model.addLayer(new DenseLayer(15, Constants::visualLatentSize, Activation::TANH));
}

void Simulation::buildAP(NNModel& model) {
	// Simple test arhitecture
	model.addLayer(new DenseLayer(Constants::visualLatentSize * 4 + 4 + 1 + 1, 20, Activation::SIGMOID));
	model.addLayer(new DenseLayer(20, 20, Activation::SIGMOID));
	model.addLayer(new DenseLayer(20, 4 + 1 + 1 + 1 + 1, Activation::SOFTMAX));
}

Simulation::Simulation() {

	Random::setSeed(Constants::seed);

	buildSIE(SIE_Network);
	buildAP(AP_Netowrk);

	SIE_Manager = NNAgentModelManager(SIE_Network, curandManager);
	AP_Manager = NNAgentModelManager(AP_Netowrk, curandManager);

	randomPositionGenerator = RndPosGenerator(Position(0, 0), Position(Constants::mapSize, Constants::mapSize));

	Tensor nullSpecieSignal = Tensor(Size(1, Constants::spicieSignalCount));
	nullSpecieSignal.initZero();
	specieSignalDict[NULL_ID] = nullSpecieSignal;

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		foodMap[i] = Constants::initialFood;
		specieMap[i] = NULL_ID;
		signalMap[i] = 0;
	}
	
	gpuUploadMaps();

}

void Simulation::addNewAgent() {

	// TODO: Signal initialization

	if (agents.size() == Constants::mapSize) return;

	Position pos;

	do {
		pos = randomPositionGenerator.next();
	} while (positionOccupied(pos));

	AgentResourceID id = getAgentID();
	SpecieID specieId = getSpecieID();
	
	SIE_Manager.registerAgent(id);
	SIE_Manager.registerNewSpiece(
		specieId,
		-Constants::SIE_InitAmplitude,
		Constants::SIE_InitAmplitude
	);

	AP_Manager.registerAgent(id);
	AP_Manager.registerNewSpiece(
		specieId,
		Constants::AP_InitializedInputs,
		Constants::AP_InitializedHidden,
		-Constants::AP_InitAmplitude,
		Constants::AP_InitAmplitude
	);

	Agent newAgent = { specieId,id,pos,pos,0 };

	specieInstaceCounter[specieId] = 1;

	agents.push_back(newAgent);
	xPositions.push_back(newAgent.pos.x);
	yPositions.push_back(newAgent.pos.y);
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;

}

bool Simulation::addAgent(Agent parent) {

	// TODO: Signal mutation

	Position pos = parent.lastPos;

	if (positionOccupied(pos)) {

		// Algorithm for random position selection around the parent agent

		bool l = positionOccupied(parent.pos + Position::left);
		bool r = positionOccupied(parent.pos + Position::right);
		bool d = positionOccupied(parent.pos + Position::down);
		bool u = positionOccupied(parent.pos + Position::up);

		unsigned short totalDirs = l + r + d + u;

		if (totalDirs == 0) return false; // No place is avalabile

		bool dirs[4] = { l,r,d,u };

		unsigned short dir = (Random::randomInt() % totalDirs) + 1;

		unsigned short i;

		for (i = 0; dir != 0; i++) {
			dir -= dirs[i];
		}

		Position deltas[4] = { Position::left, Position::right, Position::up, Position::down };

		pos = parent.pos + deltas[i];

	}

	AgentResourceID id = getAgentID();
	SpecieID specieId;
	size_t generation = parent.generation;

	SIE_Manager.registerAgent(id);
	AP_Manager.registerAgent(id);

	bool mutate = Random::runProbability(Constants::agentMutationProbability);

	if (mutate) {
		specieId = getSpecieID();

		SIE_Manager.registerSpecie(
			parent.specieId,
			specieId,
			Constants::SIE_MutationProb,
			-Constants::SIE_MutatuinAmplitude,
			Constants::SIE_MutatuinAmplitude,
			Constants::SIE_ZeroMutationProb
		);

		AP_Manager.registerSpecie(
			parent.specieId,
			specieId,
			Constants::AP_MutationProb,
			-Constants::AP_MutationAmplitude,
			Constants::AP_MutationAmplitude,
			Constants::AP_ZeroMutationProb
		);
		
		specieInstaceCounter[specieId] = 1;
		generation++;
	}
	else {
		specieId = parent.specieId;
		specieInstaceCounter[specieId]++;
	}

	Agent newAgent = { specieId,id,pos,pos,generation };

	agents.push_back(newAgent);
	xPositions.push_back(newAgent.pos.x);
	yPositions.push_back(newAgent.pos.y);
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;

	return true;
}

void Simulation::removeAgent(size_t index) {
	Agent removed = agents[index];
	agents[index] = agents[agents.size() - 1];
	agents.pop_back();
	xPositions[index] = xPositions[xPositions.size() - 1];
	xPositions.pop_back();
	yPositions[index] = yPositions[yPositions.size() - 1];
	yPositions.pop_back();

	SIE_Manager.eraseAgent(removed.id);
	AP_Manager.eraseAgent(removed.id);

	if (--specieInstaceCounter[removed.specieId] == 0) {
		SIE_Manager.eraseSpecie(removed.specieId);
		AP_Manager.eraseSpecie(removed.specieId);
		specieInstaceCounter.erase(removed.specieId);
	}

	avalabileAgentIDs.push_back(removed.id);
}

AgentResourceID Simulation::getAgentID() {
	if (avalabileAgentIDs.empty()) return lastAgentID++;
	else {
		AgentResourceID id = avalabileAgentIDs[avalabileAgentIDs.size() - 1];
		avalabileAgentIDs.pop_back();
		return id;
	}
}

SpecieID Simulation::getSpecieID() {
	return specieCounter++;
}

bool Simulation::positionOccupied(Position pos) {
	pos.wrapPositive(Constants::mapSize, Constants::mapSize);
	return specieMap[pos.y + pos.x * Constants::mapSize] != NULL_ID;
}

void Simulation::gpuUploadMaps() {
	gpuFoodMap.setValue((Tensor_HOST)foodMap);
	gpuSignalMap.setValue((Tensor_HOST)signalMap);
	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		gpuSpecieSignalMap.setRef(i, specieSignalDict[specieMap[i]]);
	}
	gpuSpecieSignalMap.syncMap();
	if (agents.size() != 0) { // If there are 0 agents then there is no array
		SIE_Manager.compile(&agents[0], agents.size());
		AP_Manager.compile(&agents[0], agents.size());
	}
	// TODO: input compile and cuda kernels
	// TODO: SIE to AP input copy kernel
}

bool Simulation::moveAgent(size_t index, Position delta) {
	Position to = agents[index].pos + delta;

	to.wrapPositive(Constants::mapSize, Constants::mapSize);

	if (!positionOccupied(to)) {

		agents[index].lastPos = agents[index].pos;
		agents[index].pos = to;
		xPositions[index] = to.x;
		yPositions[index] = to.y;

		return true;
	}
	else {
		return false;
	}
}

void Simulation::update() {

	size_t remaining = agents.size();

	while (remaining > Constants::nnPoolSize) {

		remaining -= Constants::nnPoolSize;
	}
	
	// TODO: AAAAAAAAAAAAAAAAAAAa
}

void Simulation::processDecision(size_t index, float decision[]) {
}
