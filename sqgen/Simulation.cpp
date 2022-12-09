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
	
	gpuCompile();

}

void Simulation::addNewAgent() {

	if (agents.size() == Constants::mapSize) return;

	Position pos;

	do {
		pos = randomPositionGenerator.next();
	} while (positionOccupied(pos));

	AgentID id = getAgentID();
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
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;

}

void Simulation::addAgent(Agent parent) {

	Position pos = parent.lastPos;

	if (positionOccupied(pos)) {
		// TODO: Maybe do something else ? (copy algorithm from old)
	}

	AgentID id = getAgentID();
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
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;

}

void Simulation::removeAgent(size_t index) {
	Agent removed = agents[index];
	agents[index] = agents[agents.size() - 1];
	agents.pop_back();

	SIE_Manager.eraseAgent(removed.id);
	AP_Manager.eraseAgent(removed.id);

	if (--specieInstaceCounter[removed.specieId] == 0) {
		SIE_Manager.eraseSpecie(removed.specieId);
		AP_Manager.eraseSpecie(removed.specieId);
		specieInstaceCounter.erase(removed.specieId);
	}

	avalabileAgentIDs.push_back(removed.id);
}

AgentID Simulation::getAgentID() {
	if (avalabileAgentIDs.empty()) return lastAgentID++;
	else {
		AgentID id = avalabileAgentIDs[avalabileAgentIDs.size() - 1];
		avalabileAgentIDs.pop_back();
		return id;
	}
}

SpecieID Simulation::getSpecieID() {
	return specieCounter++;
}

bool Simulation::positionOccupied(Position pos) {
	return specieMap[pos.y + pos.x * Constants::mapSize] != NULL_ID;
}

void Simulation::gpuCompile() {
	gpuFoodMap.setValue((Tensor_HOST)foodMap);
	gpuSignalMap.setValue((Tensor_HOST)signalMap);
	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		gpuSpecieSignalMap.setRef(i, specieSignalDict[specieMap[0]]);
	}
	gpuSpecieSignalMap.syncMap();
	if (agents.size() != 0) { // If there are 0 agents then there is no array
		SIE_Manager.compile(&agents[0], agents.size());
		AP_Manager.compile(&agents[0], agents.size());
	}
	// TODO: input compile and cuda kernels
	// TODO: SIE to AP input copy kernel
}

void Simulation::moveAgent(Agent& agent, Position delta) {
	Position to = agent.pos + delta;

	to.wrapPositive(Constants::mapSize, Constants::mapSize);

	if (!positionOccupied(to)) {
		agent.lastPos = agent.pos;
		agent.pos = to;
	}
}

void Simulation::update() {
	// TODO: AAAAAAAAAAAAAAAAAAAa
}