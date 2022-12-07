#include "Simulation.hpp"

#include "../ha_models/layers/DenseLayer.hpp"
#include "../ha_models/layers/SimpleRecurrentLayer.hpp"

void Simulation::buildSIE(NNModel& model) {
	// Simple test arhitecture
	model.addLayer(new DenseLayer(Constants::spicieSignalCount, 15, Activation::ReLU));
	model.addLayer(new DenseLayer(15, 15, Activation::ReLU));
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

	Agent newAgent = { specieId,id,pos,0 };

	specieInstaceCounter[specieId] = 1;

	agents.push_back(newAgent);

}

void Simulation::addAgent(Agent parent) {

	Position pos; // MUL LOGIC

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

	Agent newAgent = { specieId,id,pos,generation };

	agents.push_back(newAgent);

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

void Simulation::update() {
	// TODO: AAAAAAAAAAAAAAAAAAAa
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

	for (size_t i = 0; i < agents.size(); i++) if (agents[i].pos == pos) return true;

	return false;
}
