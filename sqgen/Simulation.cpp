#include "Simulation.hpp"

#include "../ha_models/layers/DenseLayer.hpp"
#include "../ha_models/layers/SimpleRecurrentLayer.hpp"
#include "cuda_kernels/SqGenKernels.cuh"

#include <algorithm>
#include <string>
#include <ppl.h>

int aa = 0;

/*
* 1) Sort by probability
* 2) Run actions, decrease if enough set to 0 otherwise
* 3) Erase dead agents
*/

#define COMPILE_ROUTINE 0
#define SIE_INPUT_ROUTINE 1
#define APSG_INPUT_ROUTINE 2
#define SIE_PREDICT_ROUTINE 3
#define AP_PREDICT_ROUTINE 4
#define SG_PREDICT_ROUTINE 5
#define DECISION_PROCESS_ROUTINE 6

void Simulation::buildSIE(NNModel& model) {
	model.disableDefInternalAlloc();
	model.addLayer(new DenseLayer(Constants::spicieSignalCount * 2, 10, Activation::TANH));
	model.addLayer(new DenseLayer(10, 10, Activation::TANH));
	model.addLayer(new DenseLayer(10, Constants::visualLatentSize, Activation::TANH));
}

void Simulation::buildSG(NNModel& model) {
	model.disableDefInternalAlloc();
	model.addLayer(new DenseLayer(Constants::visualLatentSize * 4 + 4 + 1 + 1, 3, Activation::TANH));
	model.addLayer(new DenseLayer(3, 3, Activation::TANH));
	model.addLayer(new DenseLayer(3, 1, Activation::TANH));
}

void Simulation::buildAP(NNModel& model) {
	model.disableDefInternalAlloc();
	model.addLayer(new DenseLayer(Constants::visualLatentSize * 4 + 4 + 1 + 1, 10, Activation::TANH));
	model.addLayer(new SimpleRecurrentLayer(10, 10, TANH, TANH));
	model.addLayer(new DenseLayer(10, 10, Activation::TANH));
	model.addLayer(new DenseLayer(10, 9, Activation::SOFTMAX));
}

Simulation::Simulation() {

	// Generate helper gpu map for processing spatial data

	decisionOutput.resize(Constants::nnPoolSize * 9);
	generatedSignalsOutput.resize(Constants::nnPoolSize);

	short observeLogic[Constants::agentObserveRangeTotal];

	for (int y = 0; y < Constants::agentObserveRange * 2 + 1; y++) {
		for (int x = 0; x < Constants::agentObserveRange * 2 + 1; x++) {
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

	float distanceLogic[Constants::agentObserveRangeTotal];

	for (int y = 0; y < Constants::agentObserveRange * 2 + 1; y++) {
		for (int x = 0; x < Constants::agentObserveRange * 2 + 1; x++) {
			int maxDist = std::max(std::abs(x - Constants::agentObserveRange),std::abs(y - Constants::agentObserveRange));
			distanceLogic[y + x * (Constants::agentObserveRange * 2 + 1)] = 1 - (1.0f / Constants::agentObserveRange) * (maxDist - 1);
		}
	}

	logicMapDistanceRange.setValue(distanceLogic);

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
		foodMap[i] = Constants::FinitialMapFood;
		rationalMapFood[i] = Constants::initialMapFood;
		specieMap[i] = NULL_ID;
		indexMap[i] = 0;
		signalMap[i] = 0;
		mediumMap[i] = Constants::mediumInitial;
	}
	
	gpuUploadMaps();

}

size_t Simulation::getAgentAt(Position pos) {
	pos.wrapPositive(Constants::mapSize, Constants::mapSize);
	return indexMap[pos.y + pos.x * Constants::mapSize];
}

void Simulation::addAgentFromSpecie(SpecieID sid, Position pos) {
	if (agents.size() == Constants::totalMapSize) return;

	AgentResourceID id = getAgentID();
	SpecieID specieId = sid;

	registerNewSpecieMember(specieId);

	// Register and allocate specie and agent resources

	SIE_Manager.registerAgent(id);
	SG_Manager.registerAgent(id);
	AP_Manager.registerAgent(id);

	Agent newAgent = { specieId,id,pos,pos,0,Constants::initialFood,Constants::agentLifetime };

	agents.push_back(newAgent); // Agent vec
	xPositions.push_back(newAgent.pos.x); // x vec
	yPositions.push_back(newAgent.pos.y); // y vec
	foodLevels.push_back(newAgent.food.toFloat());
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId; // specieMap
	indexMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = agents.size() - 1;

}

void Simulation::addNewAgent() {

	if (agents.size() == Constants::totalMapSize) return;

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

	Agent newAgent = { specieId,id,pos,pos,0,Constants::initialFood,Constants::agentLifetime };

	agents.push_back(newAgent); // Agent vec
	xPositions.push_back(newAgent.pos.x); // x vec
	yPositions.push_back(newAgent.pos.y); // y vec
	foodLevels.push_back(newAgent.food.toFloat());
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId; // specieMap
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

	Agent newAgent = { specieId,id,pos,pos,generation,Constants::initialFood,Constants::agentLifetime };

	spillFood(pos, Constants::multiplyEnergyCost - Constants::initialFood);

	agents.push_back(newAgent);
	xPositions.push_back(newAgent.pos.x);
	yPositions.push_back(newAgent.pos.y);
	foodLevels.push_back(newAgent.food.toFloat());
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;
	indexMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = agents.size() - 1;

	return true;
}

void Simulation::removeAgent(size_t index) {
	Agent removed = agents[index];

	spillFood(removed.pos, removed.food);

	// Non index based maps can be updated immediatly
	signalMap[removed.pos.y + removed.pos.x * Constants::mapSize] = 0;		 // These two make the agent invisible
	specieMap[removed.pos.y + removed.pos.x * Constants::mapSize] = NULL_ID; // to the simulation

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

	if (index == agents.size()) return;

	Position pos = agents[index].pos;
	indexMap[pos.y + pos.x * Constants::mapSize] = index;
}

SpecieID Simulation::loadSpecie(const char* path) {
	SpecieID id = getSpecieID();

	specieConrelationMap[id] = NULL;

	std::string spath(path);

	Tensor specieSignal;

	specieSignal.load((spath + "/signal.npy").c_str());

	SIE_Manager.loadSpiecie((spath + "/SIE").c_str(),id);
	AP_Manager.loadSpiecie((spath + "/AP").c_str(),id);
	SG_Manager.loadSpiecie((spath + "/SG").c_str(), id);

	specieInstaceCounter[id] = 0;
	specieSignalDict[id] = specieSignal;

	return SpecieID();
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
		//	Constants::SG_InitDetails.initializedInputs,
		//	Constants::SG_InitDetails.initializedHidden,
			-Constants::SG_InitDetails.initAmplitude,
			Constants::SG_InitDetails.initAmplitude
		);

		// AP init
		AP_Manager.registerNewSpiece(
			id,
		//	Constants::AP_InitDetails.initializedInputs,
		//	Constants::AP_InitDetails.initializedHidden,
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

void Simulation::addToAgentFood(size_t index, Rational food) {
	agents[index].food += food;
	foodLevels[index] = agents[index].food.toFloat();
}

void Simulation::setAgentPos(size_t index, Position newPos) {
	specieMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize] = 0;
	signalMap[newPos.y + newPos.x * Constants::mapSize] = signalMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize];
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
		//if (Random::runProbability(0.2f)) {
		//}
		Position front = to + delta;

		Rational displaced = getMediumAt(to);

		front.wrapPositive(Constants::mapSize, Constants::mapSize);
		if (!positionOccupied(front)) {
			setFoodAt(front, getFoodAt(to) + getFoodAt(front));
			setMediumAt(front, getMediumAt(to) + getMediumAt(front));
		}
		else {
			setFoodAt(agents[index].pos, getFoodAt(to) + getFoodAt(agents[index].pos));
			setMediumAt(agents[index].pos, getMediumAt(to) + getMediumAt(agents[index].pos));
		}

		setFoodAt(to, { 0,1 });
		setMediumAt(to, { 0,1 });
		setAgentPos(index, to);

		Rational cost = Constants::moveEnergyCost * (long long)displaced.toFloat();

		if (agents[index].food >= cost) {
			spillFood(agents[index].pos, cost);
			addToAgentFood(index, -cost);
		}
		else {
			spillFood(agents[index].pos, agents[index].food);
			addToAgentFood(index, -agents[index].food);
		}
	}
}

void Simulation::spillFood(Position pos, Rational amount) { 

	setFoodAt(pos, getFoodAt(pos) + amount);

}

void Simulation::eat(size_t index) {
	for (int i = 0; i < 4; i++) {
		Position deltas[4] = { Position::left, Position::right, Position::up, Position::down };
		Position pos = agents[index].pos + deltas[i];
		pos.wrapPositive(Constants::mapSize, Constants::mapSize);
		Rational oneOverFour = { 1,4 };
		if (getFoodAt(pos) >= Constants::eatAmount * oneOverFour) {
			addToAgentFood(index, Constants::eatAmount * oneOverFour);
			setFoodAt(pos, getFoodAt(pos) - Constants::eatAmount * oneOverFour);
		}
		else {
			addToAgentFood(index, getFoodAt(pos));
			setFoodAt(pos, { 0,1 });
		}
	}
	//Position pos = agents[index].pos + agents[index].pos - agents[index].lastPos;
	//pos.wrapPositive(Constants::mapSize, Constants::mapSize);
	//if (getFoodAt(pos) >= Constants::eatAmount) {
	//	addToAgentFood(index, Constants::eatAmount);
	//	setFoodAt(pos, getFoodAt(pos) - Constants::eatAmount);
	//}
	//else {
	//	addToAgentFood(index, getFoodAt(pos));
	//	setFoodAt(pos, { 0,1 });
	//}
}

void Simulation::attack(size_t index) {

	//Rational totalFoodBalance = {0,1};
	//totalFoodBalance -= Constants::attackEnergyCost;
	//addToAgentFood(index, -Constants::attackEnergyCost);
	//spillFood(agents[index].pos, Constants::attackEnergyCost);

	//for (int i = 0; i < 4; i++) {
	//	Position pos = agents[index].pos  + (agents[index].pos - agents[index].lastPos);
	//	pos.wrapPositive(Constants::mapSize, Constants::mapSize);
	//	if (positionOccupied(pos)) {
	//		size_t target = getAgentAt(pos);
	//		if (Constants::attackEnergyGain <= agents[target].food) {
	//			//totalFoodBalance += Constants::attackEnergyGain;
	//			spillFood(pos, Constants::attackEnergyGain);
	//			addToAgentFood(target, -Constants::attackEnergyGain);
	//		}
	//		else {
	//			//totalFoodBalance += agents[target].food;
	//			spillFood(pos, agents[target].food);
	//			addToAgentFood(target, -agents[target].food);
	//		}
	//	}

	//}

	Position deltas[4] = { Position::left, Position::right, Position::up, Position::down };

	Rational totalFoodBalance = { 0,1 };

	for (int i = 0; i < 4; i++) {
		Position pos = agents[index].pos + deltas[i];
		pos.wrapPositive(Constants::mapSize, Constants::mapSize);
		if (positionOccupied(pos)) {
			size_t target = getAgentAt(pos);
			if (Constants::attackEnergyGain <= agents[target].food) {
				totalFoodBalance += Constants::attackEnergyGain;
				addToAgentFood(target, -Constants::attackEnergyGain);
			}
			else {
				totalFoodBalance += agents[target].food;
				addToAgentFood(target, -agents[target].food);
			}
		}

	}

	if (totalFoodBalance == Rational()) {
		if (agents[index].food >= Constants::attackEnergyCost) {
			spillFood(agents[index].pos, Constants::attackEnergyCost);
			addToAgentFood(index, -Constants::attackEnergyCost);
		}
		else {
			spillFood(agents[index].pos, agents[index].food);
			addToAgentFood(index, -agents[index].food);
		}
	}
	else {
		addToAgentFood(index, totalFoodBalance);
	}
}

void Simulation::share(size_t index) {

	Rational oneOverFour = { 1,4 };
	Rational sharable = Constants::shareEnergyTransfer * oneOverFour;

	Position pos = agents[index].pos;

	Position deltas[4] = { Position::left, Position::right, Position::up, Position::down };

	for (int i = 0; i < 4; i++) {
		Position relative = pos + deltas[i];
		relative.wrapPositive(Constants::mapSize, Constants::mapSize);
		if (!positionOccupied(relative)) {
			if (agents[index].food < sharable) sharable = agents[index].food;
			spillFood(relative, sharable);
			addToAgentFood(index, -sharable);
			if (agents[index].food == Rational()) return;
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
	SIE_InputPool.initZero();
	gpuSync();
	SqGenKernels::processSIEInputs(
		logicMapObserveRange.getGpuPointer(),
		logicMapDistanceRange.getGpuPointer(),
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


	profiler.start(APSG_INPUT_ROUTINE);

	APSG_InputPool.initZero();
	gpuSync();
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

int maxi(float* v, int size) {
	float prev = 0;
	int iprev = 0;
	for (int i = 0; i < size; i++) {
		if (prev < v[i]) {
			prev = v[i];
			iprev = i;
		}
	}
	return iprev;
}

void Simulation::runAPSGAndProcessDecisions(size_t from, size_t to) {
	Tensor slicedAPSG = APSG_InputPool.slice(0, to - from);
	// What the fuck

	gpuSync();

	profiler.start(SG_PREDICT_ROUTINE);
	Tensor generatedSignals = SG_Manager.predict(slicedAPSG);
	profiler.end(SG_PREDICT_ROUTINE);

	gpuSync();

	profiler.start(AP_PREDICT_ROUTINE);
	Tensor decisions = AP_Manager.predict(slicedAPSG);
	profiler.end(AP_PREDICT_ROUTINE);

	gpuSync();

	decisions.getValue(&decisionOutput[from*9]);
	generatedSignals.getValue(&generatedSignalsOutput[from]);

}

void Simulation::update() {

	profiler.reset();

	if (paused && !step) return;
	if (step) step = !step;


	gpuSync();

	gpuUploadMaps();

	size_t remaining = agents.size();
	size_t current = 0;

	for (int i = 0; i < 9; i++) actionTracker[i] = 0;

	while(agents.size() * 9 >= decisionOutput.size()) decisionOutput.resize(2 * decisionOutput.size());
	while (agents.size() >= generatedSignalsOutput.size()) generatedSignalsOutput.resize(2 * generatedSignalsOutput.size());

	while (remaining > Constants::nnPoolSize - 500) {
		pipelineToAPSG(current, current + Constants::nnPoolSize - 500);
		runAPSGAndProcessDecisions(current, current + Constants::nnPoolSize - 500);
		remaining -= Constants::nnPoolSize - 500;
		current += Constants::nnPoolSize - 500;
	}

	if (remaining != 0) {
		pipelineToAPSG(current, current + remaining);
		runAPSGAndProcessDecisions(current, current + remaining);
	}

	for (size_t x = 0; x < Constants::mapSize; x++) {
		for (size_t y = 0; y < Constants::mapSize; y++) {
			Position pos(x, y);
			//spillFood(pos, { 1,100 });
			std::random_shuffle(deltas.begin(), deltas.end());
			for (int i = 0; i < 4; i++) {
				Position dp = (pos + deltas[i]);
				if (positionOccupied(dp)) continue;
				dp.wrapPositive(Constants::mapSize, Constants::mapSize);
				if (getFoodAt(dp) < getFoodAt(pos)) {
					Rational f = getFoodAt(pos) - getFoodAt(dp);
					if (f.b < 100) f.multiply(100);
					f.a /= 25;
					setFoodAt(dp, getFoodAt(dp) + f);
					setFoodAt(pos, getFoodAt(pos) - f);
				}
				if (getMediumAt(dp) < getMediumAt(pos)) {
					Rational f = getMediumAt(pos) - getMediumAt(dp);
					if (f.b < 100) f.multiply(100);
					f.a /= 25;
					setMediumAt(dp, getMediumAt(dp) + f);
					setMediumAt(pos, getMediumAt(pos) - f);
				}
				else continue;
			}

		}
	}

	profiler.start(DECISION_PROCESS_ROUTINE);

	size_t initialSize = agents.size();

	for (size_t i = 0; i < initialSize; i++) {
		//if (generatedSignalsOutput[i] != generatedSignalsOutput[i]) {
		//	__debugbreak();
		//	std::vector<float> a(slicedAPSG.size.size);
		//	slicedAPSG.getValue(a.data());
		//	Tensor* ts = SG_Manager.agentModelVariables[agents[i].specieId];// 9 to 9 error
		//	std::vector<Layer*> layers = SG_Manager.supermodel.layers;
		//	for (int l = 0; l < layers.size(); l++) {
		//		Tensor t = layers[l]->getValue();
		//		std::vector<float> ff(t.size.size);
		//		t.getValue(ff.data());
		//	}
		//	for (int l = 0; l < SG_Manager.paramCount; l++) {
		//		std::vector<float> f(ts[l].size.size);
		//		ts[l].getValue(f.data());
		//	}
		//	slicedAPSG.save("input.npy");
		//	ts[0].save("params.npy");
		//	ts[1].save("biases.npy");
		//}
		Agent& agent = agents[i];
		Position pos = agent.pos;
		int action = Random::runProbabilityVector(&decisionOutput[i * 9], 9);
		actionTracker[action]++;
		//if (!(generatedSignalsOutput[i] != generatedSignalsOutput[i]))signalMap[pos.y + pos.x * Constants::mapSize] = generatedSignalsOutput[i];
		//else {
		//	signalMap[pos.y + pos.x * Constants::mapSize] = 0;
		//	aa++;
		//}
		switch (action) {
		case EAT:
			eat(i);
			break;
		case MULTIPLY:
			if (agent.food > Constants::multiplyEnergyCost) {
				if (addAgent(agent)) addToAgentFood(i, -Constants::multiplyEnergyCost);
			}
			else {
				//spillFood(agent.pos, agent.food);
				//addToAgentFood(i, -agent.food);
			}
			break;
		case UP:
			moveAgent(i, Position(0, -1));
			break;
		case DOWN:
			moveAgent(i, Position(0, 1));
			break;
		case LEFT:
			moveAgent(i, Position(-1, 0));
			break;
		case RIGHT:
			moveAgent(i, Position(1, 0));
			break;
		case ATTACK:
			attack(i);
			break;
		case SHARE:
			share(i);
			break;
		case SIGNAL:
			if (!(generatedSignalsOutput[i] != generatedSignalsOutput[i]))signalMap[pos.y + pos.x * Constants::mapSize] = generatedSignalsOutput[i];
			else {
				signalMap[pos.y + pos.x * Constants::mapSize] = 0;
				aa++;
			}
			break;
		}
	}

	//Rational r = agents[20].food;
	profiler.end(DECISION_PROCESS_ROUTINE);
	
	std::vector<size_t> toRemove;

	for (size_t i = agents.size() - 1; i < agents.size(); i--) {
		if (agents[i].food <= Rational() || agents[i].food > Constants::maximumFood || agents[i].lifetime == 0) toRemove.push_back(i);
	}

	for (size_t i = 0; i < toRemove.size(); i++) {
		removeAgent(toRemove[i]);
	}

	printProfilerInfo();
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

	float ae = getAgenentEnergy();

	std::cout << "Total agents: " << agents.size() << "\n";

	std::cout << "Compile Time: " << profiler.get(COMPILE_ROUTINE) << " ms\n"
		<< "SIE Input: " << profiler.get(SIE_INPUT_ROUTINE) << " ms\n"
		<< "SIE Predict: " << profiler.get(SIE_PREDICT_ROUTINE) << " ms\n"
		<< "APSG Input: " << profiler.get(APSG_INPUT_ROUTINE) << " ms\n"
		<< "AP Predict: " << profiler.get(AP_PREDICT_ROUTINE) << " ms\n"
		<< "SG Predict: " << profiler.get(SG_PREDICT_ROUTINE) << " ms\n"
		<< "Decision processing: " << profiler.get(DECISION_PROCESS_ROUTINE) << " ms\n"
		<< "Total simulation food: " << getTotalFood() << '\n'
		<< "Total simulation medium: " << getTotalMedium() << '\n'
		<< "Agent energy: " <<ae << '\n'
		<< "Energy per agent: " << ae/agents.size() << '\n'
		<< aa<<'\n'
		<< "Actions: \n";

	for (int i = 0; i < 9; i++) std::cout << (float)actionTracker[i] / agents.size() << '\n';
 
}

std::map<SpecieID, Tensor>& Simulation::getSignalDict() {
	return specieSignalDict;
}

float Simulation::getTotalFood() {
	float total = 0;

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		total += rationalMapFood[i].toFloat();
	}

	for (size_t i = 0; i < agents.size(); i++) {
		total += agents[i].food.toFloat();
	}

	return total;

}

Rational Simulation::getFoodAt(Position pos) {
	return rationalMapFood[pos.y + pos.x * Constants::mapSize];
}

void Simulation::setFoodAt(Position pos, Rational r) {

	rationalMapFood[pos.y + pos.x * Constants::mapSize] = r;

	foodMap[pos.y + pos.x * Constants::mapSize] = r.toFloat();

}

void Simulation::restartFoodMap() {
	for (size_t i = 0; i < Constants::totalMapSize; i++) {

		foodMap[i] = Constants::FinitialMapFood;
		rationalMapFood[i] = Constants::initialMapFood;
	}
}

float Simulation::getAgenentEnergy() {

	float total = 0;

	for (size_t i = 0; i < agents.size(); i++) {
		total += agents[i].food.toFloat();
	}

	return total;
}

Rational Simulation::getMediumAt(Position pos) {
	return mediumMap[pos.y + pos.x * Constants::mapSize];
}

void Simulation::setMediumAt(Position pos,Rational value) {
	mediumMap[pos.y + pos.x * Constants::mapSize] = value;
}

float Simulation::getTotalMedium() {
	Rational total = { 0,1 };

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		total += mediumMap[i];
	}

	return total.toFloat();
}

Position Simulation::randomUnoccupiedPosition() {
	Position pos;

	do {
		pos = randomPositionGenerator.next();
	} while (positionOccupied(pos));

	return pos;
}

void Simulation::togglePause() {
	paused = !paused;
}