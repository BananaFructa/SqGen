#include "Simulation.hpp"

#include "../ha_models/layers/DenseLayer.hpp"
#include "../ha_models/layers/SimpleRecurrentLayer.hpp"
#include "cuda_kernels/SqGenKernels.cuh"

#include <algorithm>

#define COMPILE_ROUTINE 0
#define SIE_INPUT_ROUTINE 1
#define APSG_INPUT_ROUTINE 2
#define SIE_PREDICT_ROUTINE 3
#define AP_PREDICT_ROUTINE 4
#define SG_PREDICT_ROUTINE 5
#define DECISION_PROCESS_ROUTINE 6

void Simulation::buildSIE(NNModel& model) {
	model.disableDefInternalAlloc();
	// Simple test arhitecture
	model.addLayer(new DenseLayer(Constants::spicieSignalCount, 20, Activation::TANH));
	model.addLayer(new DenseLayer(20, 20, Activation::TANH));
	model.addLayer(new DenseLayer(20, Constants::visualLatentSize, Activation::TANH));
}

void Simulation::buildSG(NNModel& model) {
	model.disableDefInternalAlloc();
	model.addLayer(new DenseLayer(Constants::visualLatentSize * 4 + 4 + 1 + 1, 5, Activation::SIGMOID));
	model.addLayer(new DenseLayer(5, 5, Activation::SIGMOID));
	model.addLayer(new DenseLayer(5, 1, Activation::TANH));
}

void Simulation::buildAP(NNModel& model) {
	model.disableDefInternalAlloc();
	// Simple test arhitecture
	model.addLayer(new DenseLayer(Constants::visualLatentSize * 4 + 4 + 1 + 1, 20, Activation::SIGMOID));
	model.addLayer(new DenseLayer(20, 20, Activation::SIGMOID));
	model.addLayer(new DenseLayer(20, 9, Activation::SOFTMAX));
}

Simulation::Simulation() {

	// Generate helper gpu map for processing spatial data

	short observeLogic[Constants::agentObserveRangeTotal];

	for (size_t y = 0; y < Constants::agentObserveRange * 2 + 1; y++) {
		for (size_t x = 0; x < Constants::agentObserveRange * 2 + 1; x++) {
			if (x < Constants::agentObserveRange && y < Constants::agentObserveRange) {
				observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = LEFT_FIRST | UP_SECOND | USE_FIRST | USE_SECOND;
			}
			if (x < Constants::agentObserveRange && y > Constants::agentObserveRange) {
				observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = LEFT_FIRST | DOWN_SECOND | USE_FIRST | USE_SECOND;
			}
			if (x > Constants::agentObserveRange && y < Constants::agentObserveRange) {
				observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = UP_FIRST | RIGHT_SECOND | USE_FIRST | USE_SECOND;
			}
			if (x > Constants::agentObserveRange && y > Constants::agentObserveRange) {
				observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = DOWN_FIRST | RIGHT_SECOND | USE_FIRST | USE_SECOND;
			}
			if (x == Constants::agentObserveRange) {
				if (y < Constants::agentObserveRange)
					observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = UP_FIRST | USE_FIRST;
				if (y > Constants::agentObserveRange)
					observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = DOWN_FIRST | USE_FIRST;
			}
			if (y == Constants::agentObserveRange) {
				if (x < Constants::agentObserveRange)
					observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = LEFT_FIRST | USE_FIRST;
				if (x > Constants::agentObserveRange)
					observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = RIGHT_FIRST | USE_FIRST;
			}
			if (x == Constants::agentObserveRange && y == Constants::agentObserveRange)
				observeLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = 0;
		}
	}

	logicMapObserveRange.setValue(observeLogic);

	// =====================================================

	Random::setSeed(Constants::seed);

	buildSIE(SIE_Network);
	buildSG(SG_Network);
	buildAP(AP_Netowrk);

	SIE_Manager = NNAgentModelManager(SIE_Network, curandManager);
	SG_Manager = NNAgentModelManager(SG_Network, curandManager);
	AP_Manager = NNAgentModelManager(AP_Netowrk, curandManager);

	randomPositionGenerator = RndPosGenerator(Position(0, 0), Position(Constants::mapSize, Constants::mapSize));

	Tensor nullSpecieSignal = Tensor(Size(1, Constants::spicieSignalCount));
	nullSpecieSignal.initZero();
	specieSignalDict[NULL_ID] = nullSpecieSignal;

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		foodMap[i] = Constants::initialMapFood;
		specieMap[i] = NULL_ID;
		indexMap[i] = 0;
		signalMap[i] = 0;
	}
	
	gpuUploadMaps();

}

size_t Simulation::getAgentAt(Position pos) {
	return indexMap[pos.y + pos.x * Constants::mapSize];
}

void Simulation::addNewAgent() {

	if (agents.size() == Constants::mapSize) return;

	// Choose position

	Position pos;

	do {
		pos = randomPositionGenerator.next();
	} while (positionOccupied(pos));

	AgentResourceID id = getAgentID();
	SpecieID specieId = newSpiecie(NULL_ID);

	registerNewSpecieMember(specieId);

	// Register and allocate specie and agent resources
	
	SIE_Manager.registerAgent(id);
	SG_Manager.registerAgent(id);
	AP_Manager.registerAgent(id);

	// Create agent and update all necessart registries

	Agent newAgent = { specieId,id,pos,pos,0,Constants::initialFood };

	agents.push_back(newAgent); // Agent vec
	xPositions.push_back(newAgent.pos.x); // x vec
	yPositions.push_back(newAgent.pos.y); // y vec
	foodLevels.push_back(newAgent.food);
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId; // specieMap
	size_t r = agents.size() - 1;
	indexMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = agents.size() - 1;

}

bool Simulation::addAgent(Agent parent) {

	Position pos = parent.lastPos;

	if (positionOccupied(pos)) {

		// Algorithm for random position selection around the parent agent

		bool l = !positionOccupied(parent.pos + Position::left);
		bool r = !positionOccupied(parent.pos + Position::right);
		bool d = !positionOccupied(parent.pos + Position::down);
		bool u = !positionOccupied(parent.pos + Position::up);

		unsigned short totalDirs = l + r + d + u;

		if (totalDirs == 0) {
			return false;
		}// No place is avalabile

		bool dirs[4] = { l,r,u,d };

		unsigned short dir = (Random::randomInt() % totalDirs) + 1;

		unsigned short i;

		for (i = 0; dir != 0; i++) {
			dir -= dirs[i];
		}

		i--;

		Position deltas[4] = { Position::left, Position::right, Position::up, Position::down };

		pos = parent.pos + deltas[i];
		pos.wrapPositive(Constants::mapSize, Constants::mapSize);

	}

	AgentResourceID id = getAgentID();
	SpecieID specieId;
	size_t generation = parent.generation;

	SIE_Manager.registerAgent(id);
	SG_Manager.registerAgent(id);
	AP_Manager.registerAgent(id);

	bool mutate = Random::runProbability(Constants::agentMutationProbability);

	if (mutate) {
		specieId = newSpiecie(parent.specieId);
		generation++;
	}
	else {
		specieId = parent.specieId;
	}

	registerNewSpecieMember(specieId);

	Agent newAgent = { specieId,id,pos,pos,generation,Constants::initialFood };

	agents.push_back(newAgent);
	xPositions.push_back(newAgent.pos.x);
	yPositions.push_back(newAgent.pos.y);
	foodLevels.push_back(newAgent.food);
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;
	indexMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = agents.size() - 1;

	return true;
}

void Simulation::removeAgent(size_t index) {
	if (!std::count(removePendingList.begin(), removePendingList.end(), index)) {
		removePendingList.push_back(index);
		Agent removed = agents[index];

		// Non index based maps can be updated immediatly
		signalMap[removed.pos.y + removed.pos.x * Constants::mapSize] = 0;
		specieMap[removed.pos.y + removed.pos.x * Constants::mapSize] = NULL_ID;
	}
}

void Simulation::processRemoveRequests() {
	for (size_t i = 0; i < removePendingList.size(); i++) {
		size_t index = removePendingList[i];
		Agent removed = agents[index];
		agents[index] = agents[agents.size() - 1];
		agents.pop_back();
		xPositions[index] = xPositions[xPositions.size() - 1];
		xPositions.pop_back();
		yPositions[index] = yPositions[yPositions.size() - 1];
		yPositions.pop_back();
		foodLevels[index] = foodLevels[foodLevels.size() - 1];
		foodLevels.pop_back();

		SIE_Manager.eraseAgent(removed.id);
		SG_Manager.eraseAgent(removed.id);
		AP_Manager.eraseAgent(removed.id);

		eraseSpecieMember(removed.specieId);

		avalabileAgentIDs.push_back(removed.id);
	}

	removePendingList.clear();
}

SpecieID Simulation::newSpiecie(size_t parent) {
	SpecieID id = getSpecieID();

	Tensor specieSignal = specieSignalAllocator.getTensor();

	specieConrelationMap[id] = parent;

	if (parent == NULL_ID) {

		// SIE init
		SIE_Manager.registerNewSpiece(
			id,
			-Constants::SIE_InitDetails.initAmplitude,
			Constants::SIE_InitDetails.initAmplitude
		);

		SG_Manager.registerNewSpiece(
			id,
			Constants::SG_InitDetails.initializedInputs,
			Constants::SG_InitDetails.initializedHidden,
			-Constants::SG_InitDetails.initAmplitude,
			Constants::SG_InitDetails.initAmplitude
		);

		// AP init
		AP_Manager.registerNewSpiece(
			id,
			Constants::AP_InitDetails.initializedInputs,
			Constants::AP_InitDetails.initializedHidden,
			-Constants::AP_InitDetails.initAmplitude,
			Constants::AP_InitDetails.initAmplitude
		);

		// Specie signal gen
		curandManager.randomizeTensorUniform(
			specieSignal,
			-Constants::specieSignalAmplitude,
			Constants::specieSignalAmplitude
		);

	}
	else {
		SIE_Manager.registerSpecie(
			parent,
			id,
			Constants::SIE_MutationDetails.mutationProbability,
			-Constants::SIE_MutationDetails.mutationAmplitude,
			Constants::SIE_MutationDetails.mutationAmplitude,
			Constants::SIE_MutationDetails.zeroMutationProbability
		);
		
		SG_Manager.registerSpecie(
			parent,
			id,
			Constants::SG_MutationDetails.mutationProbability,
			-Constants::SG_MutationDetails.mutationAmplitude,
			Constants::SG_MutationDetails.mutationAmplitude,
			Constants::SG_MutationDetails.zeroMutationProbability
		);

		AP_Manager.registerSpecie(
			parent,
			id,
			Constants::AP_MutationDetails.mutationProbability,
			-Constants::AP_MutationDetails.mutationAmplitude,
			Constants::AP_MutationDetails.mutationAmplitude,
			Constants::AP_MutationDetails.zeroMutationProbability
		);

		specieSignalDict[parent].copyTo(specieSignal);
		curandManager.rndOffsetTensorUniform(
			specieSignal,
			Constants::specieSignalMutationProb,
			-Constants::specieSignalMutatuionAmplitude,
			Constants::specieSignalMutatuionAmplitude
		);
		gpuSync(); // <- this might be a bad idea
		specieSignal.clamp(-Constants::specieSignalAmplitude, Constants::specieSignalAmplitude);
	}

	specieInstaceCounter[id] = 0;
	specieSignalDict[id] = specieSignal;

	return id;
}

void Simulation::registerNewSpecieMember(SpecieID specie) {
	specieInstaceCounter[specie]++;
}

void Simulation::eraseSpecieMember(SpecieID specie) {
	if (--specieInstaceCounter[specie] == 0) {
		SIE_Manager.eraseSpecie(specie);
		SG_Manager.eraseSpecie(specie);
		AP_Manager.eraseSpecie(specie);
		specieInstaceCounter.erase(specie);
		specieSignalAllocator.freeTensor(specieSignalDict[specie]);
		specieSignalDict.erase(specie);
		specieConrelationMap.erase(specie);
	}
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
	for (size_t x = 0; x < Constants::mapSize; x++) {
		for (size_t y = 0; y < Constants::mapSize; y++) {
			gpuSpecieSignalMap.setRef(y + x * Constants::mapSize, specieSignalDict[specieMap[y + x * Constants::mapSize]]);
		}
	}
	gpuSpecieSignalMap.syncMap();
}

bool Simulation::addToAgentFood(size_t index, float food) {
	agents[index].food += food;
	foodLevels[index] += food;
	if (agents[index].food <= 0 || agents[index].food > Constants::maximumFood) {
		removeAgent(index);
		return false;
	}
	return true;
}

void Simulation::setAgentPos(size_t index, Position newPos) {
	specieMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize] = 0;
	signalMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize] = 0;
	agents[index].lastPos = agents[index].pos;
	agents[index].pos = newPos;
	xPositions[index] = newPos.x;
	yPositions[index] = newPos.y;
	specieMap[newPos.y + newPos.x * Constants::mapSize] = agents[index].specieId;
	indexMap[newPos.y + newPos.x * Constants::mapSize] = index;
}

void Simulation::moveAgent(size_t index, Position delta) {
	// Todo: Move signal
	Position to = agents[index].pos + delta;

	to.wrapPositive(Constants::mapSize, Constants::mapSize);

	if (!positionOccupied(to)) {

		if (!addToAgentFood(index, -Constants::moveEnergyCost)) {
			removeAgent(index);
			return;
		}

		setAgentPos(index, to);
	}
}

void Simulation::eat(size_t index) {
	Position pos = agents[index].pos;
	if (foodMap[pos.y + pos.x * Constants::mapSize] >= Constants::eatAmount) {
		if (!addToAgentFood(index, Constants::eatAmount)) removeAgent(index);
		foodMap[pos.y + pos.x * Constants::mapSize] -= Constants::eatAmount;
	}
	else {
		if (!addToAgentFood(index, foodMap[pos.y + pos.x * Constants::mapSize])) removeAgent(index);
		foodMap[pos.y + pos.x * Constants::mapSize] = 0;
	}
}

void Simulation::attack(size_t index) {

	float totalFoodBalance = -Constants::attackEnergyCost;

	for (int i = 0; i < 4; i++) {
		Position pos = agents[index].pos + dirs[i];
		pos.wrapPositive(Constants::mapSize, Constants::mapSize);
		if (positionOccupied(pos)) {
			size_t target = getAgentAt(pos);
			totalFoodBalance += std::min(Constants::attackEnergyGain,agents[target].food);
			// The max in this case is not really necessary but it's put for consistency
			if (!addToAgentFood(target, std::max(-Constants::attackEnergyGain,-agents[target].food))) removeAgent(target);
		}

	}

	if (!addToAgentFood(index, totalFoodBalance)) removeAgent(index);
}

void Simulation::share(size_t index) {

	float sharable = std::min(agents[index].food, Constants::shareEnergyTransfer);

	size_t neighbours = 0;

	Position pos = agents[index].pos;

	for (int x = -Constants::shareRadius; x <= Constants::shareRadius; x++) {
		for (int y = -Constants::shareRadius; y <= Constants::shareRadius; y++) {
			Position current = Position(x, y);
			Position relative = pos + current;
			relative.wrapPositive(Constants::mapSize, Constants::mapSize);
			if (positionOccupied(relative)) neighbours++;
		}
	}

	if (neighbours != 0) if (!addToAgentFood(index, -sharable)) removeAgent(index);

	for (int x = -Constants::shareRadius; x <= Constants::shareRadius; x++) {
		for (int y = -Constants::shareRadius; y <= Constants::shareRadius; y++) {
			Position current = Position(x, y);
			Position relative = pos + current;
			if (positionOccupied(relative)) {
				int n = getAgentAt(relative);
				if (!addToAgentFood(n, sharable / neighbours)) removeAgent(n);
			}
		}
	}
}

void Simulation::pipelineToAPSG(size_t from, size_t to) {

	profiler.start(COMPILE_ROUTINE);
	SIE_Manager.compile(&agents[from], to - from, 4);
	SG_Manager.compile(&agents[from], to - from);
	AP_Manager.compile(&agents[from], to - from);

	gpuXPositions.slice(0,to - from).setValue(&xPositions[from]);
	gpuYPositions.slice(0,to - from).setValue(&yPositions[from]);
	gpuFoodLevels.slice(0, to - from).setValue(&foodLevels[from]);
	profiler.end(COMPILE_ROUTINE);

	profiler.start(SIE_INPUT_ROUTINE);
	SqGenKernels::processSIEInputs(
		logicMapObserveRange.getGpuPointer(),
		gpuSpecieSignalMap.getGpuMapPointer(),
		gpuXPositions.getGpuPointer(),
		gpuYPositions.getGpuPointer(),
		SIE_InputPool.getGpuDataPointer(),
		Constants::agentObserveRange,
		Constants::mapSize,
		Constants::spicieSignalCount,
		to - from
	);

	gpuSync();

	profiler.end(SIE_INPUT_ROUTINE);

	Tensor slicedSIE_pool = SIE_InputPool.slice(0, (to - from) * 4);

	profiler.start(SIE_PREDICT_ROUTINE);

	Tensor SIE_out = SIE_Manager.predict(slicedSIE_pool);

	profiler.end(SIE_PREDICT_ROUTINE);

	// From here nothing is tested

	profiler.start(APSG_INPUT_ROUTINE);

	SqGenKernels::processAPSGInputs(
		logicMapObserveRange.getGpuPointer(),
		gpuXPositions.getGpuPointer(),
		gpuYPositions.getGpuPointer(),
		SIE_out.getGpuDataPointer(), // already sliced as is outputed from the NN
		gpuFoodMap.getGpuDataPointer(),
		gpuFoodLevels.getGpuPointer(),
		gpuSignalMap.getGpuDataPointer(),
		APSG_InputPool.getGpuDataPointer(),
		Constants::agentObserveRange,
		Constants::mapSize,
		to - from

	);

	gpuSync();

	profiler.end(APSG_INPUT_ROUTINE);

	// Sync and compile AP inputs
}

void Simulation::runAPSGAndProcessDecisions(size_t from, size_t to) {
	Tensor slicedAPSG = APSG_InputPool.slice(0, to - from);

	// What the fuck
	profiler.start(SG_PREDICT_ROUTINE);
	Tensor generatedSignals = SG_Manager.predict(slicedAPSG);
	profiler.end(SG_PREDICT_ROUTINE);

	profiler.start(AP_PREDICT_ROUTINE);
	Tensor decisions = AP_Manager.predict(slicedAPSG);
	profiler.end(AP_PREDICT_ROUTINE);

	gpuSync();

	profiler.start(DECISION_PROCESS_ROUTINE);
	decisions.getValue(decisionOutput);
	generatedSignals.getValue(generatedSignalsOutput);

	//for (int i = 0; i < 3; i++) {
	//	std::cout << " " << generatedSignalsOutput[i] << " \n";
	//}

	for (size_t i = 0; i < to - from; i++) {
		Agent& agent = agents[from + i];
		Position pos = agent.pos;
		switch (Random::runProbabilityVector(&decisionOutput[i * 9], 9)) {
		case EAT:
			eat(from + i);
			break;
		case MULTIPLY:
			addAgent(agent);
			break;
		case UP:
			moveAgent(from + i, Position(0, -1));
			break;
		case DOWN:
			moveAgent(from + i, Position(0, 1));
			break;
		case LEFT:
			moveAgent(from + i, Position(-1, 0));
			break;
		case RIGHT:
			moveAgent(from + i, Position(1, 0));
			break;
		case ATTACK:
			attack(from + i);
			break;
		case SHARE:
			share(from + i);
			break;
		case SIGNAL:
			signalMap[pos.y + pos.x * Constants::mapSize] = generatedSignalsOutput[i];
			break;
		}
	}
	profiler.end(DECISION_PROCESS_ROUTINE);
}

void Simulation::update() {

	profiler.reset();

	if (paused) return;

	gpuSync();

	gpuUploadMaps();

	size_t remaining = agents.size();
	size_t current = 0;

	while (remaining > Constants::nnPoolSize) {
		pipelineToAPSG(current, current + Constants::nnPoolSize);
		runAPSGAndProcessDecisions(current, current + Constants::nnPoolSize);
		remaining -= Constants::nnPoolSize;
		current += Constants::nnPoolSize;
	}

	if (remaining != 0) {
		pipelineToAPSG(current, current + remaining);
		runAPSGAndProcessDecisions(current, current + remaining);

	}

	processRemoveRequests();
	
	// TODO: AAAAAAAAAAAAAAAAAAAa
}

SpecieID Simulation::getParentSpecie(SpecieID child) {
	return specieConrelationMap[child];
}

float* Simulation::getFoodMap() {
	return foodMap;
}

SpecieID* Simulation::getSpecieMap() {
	return specieMap;
}

float* Simulation::getSignalMap() {
	return signalMap;
}

std::vector<Agent>& Simulation::getAgents() {
	return agents;
}

void Simulation::printProfilerInfo() {

	std::cout << "Total agents: " << agents.size() << "\n";

	std::cout << "Compile Time: " << profiler.get(COMPILE_ROUTINE) << " ms\n"
		<< "SIE Input: " << profiler.get(SIE_INPUT_ROUTINE) << " ms\n"
		<< "SIE Predict: " << profiler.get(SIE_PREDICT_ROUTINE) << " ms\n"
		<< "APSG Input: " << profiler.get(APSG_INPUT_ROUTINE) << " ms\n"
		<< "AP Predict: " << profiler.get(AP_PREDICT_ROUTINE) << " ms\n"
		<< "SG Predict: " << profiler.get(SG_PREDICT_ROUTINE) << " ms\n"
		<< "Decision processing: " << profiler.get(DECISION_PROCESS_ROUTINE) << " ms\n";
 
}

std::map<SpecieID, Tensor>& Simulation::getSignalDict() {
	return specieSignalDict;
}

void Simulation::togglePause() {
	paused = !paused;
}