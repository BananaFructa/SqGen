#include "Simulation.hpp"

#include "../ha_models/layers/DenseLayer.hpp"
#include "../ha_models/layers/SimpleRecurrentLayer.hpp"
#include "cuda_kernels/SqGenKernels.cuh"

#include <algorithm>
#include <string>
#include <ppl.h>

#include "highfive/H5File.hpp"


int aa = 0;

Color randomAgentColor() {
	return Color(Random::randomInt() % 196 + 60, Random::randomInt() % 196 + 60, Random::randomInt() % 196 + 60);
}

const int colorChangeAmplitude = 13;

int mutateChannel(int c) {
	if (c >= 255 - (colorChangeAmplitude - 1) / 2) {
		c -= Random::randomInt() % colorChangeAmplitude;
	}
	else if (c <= 60 + (colorChangeAmplitude - 1) / 2) {
		c += Random::randomInt() % colorChangeAmplitude;
	}
	else {
		c += Random::randomInt() % colorChangeAmplitude - (colorChangeAmplitude - 1) / 2;
	}
	return c;
}

Color mutateColor(Color color) {

	return Color(mutateChannel(color.r), mutateChannel(color.g), mutateChannel(color.b));

}

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
#define UPDATE_DELTA 7

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
	model.addLayer(new DenseLayer(Constants::visualLatentSize * 4 + 4 + 1 + 1, 5, Activation::TANH));
	//model.addLayer(new SimpleRecurrentLayer(10, 10, TANH, TANH));
	model.addLayer(new DenseLayer(5, 5, Activation::TANH));
	model.addLayer(new DenseLayer(5, 5, Activation::TANH));
	model.addLayer(new DenseLayer(5, 6, Activation::TANH));
}

Simulation::Simulation() {

	cudaMallocHost(&foodMap, Constants::totalMapSize * sizeof(float));
	cudaMallocHost(&signalMap, Constants::totalMapSize * sizeof(float));
	cudaMallocHost(&specieMap, Constants::totalMapSize * sizeof(SpecieID));

	// Generate helper gpu map for processing spatial data

	decisionOutput.resize(Constants::nnPoolSize * 6);
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

	randomPositionGenerator = RndPosGenerator(Position2i(0, 0), Position2i(Constants::mapSize, Constants::mapSize));

	Tensor nullSpecieSignal = Tensor(Size(1, Constants::spicieSignalCount));
	nullSpecieSignal.initZero();
	specieSignalDict[NULL_ID] = nullSpecieSignal;

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		foodMap[i] = Constants::FinitialMapFood;
		rationalMapFood[i] = Constants::initialMapFood;
		specieMap[i] = NULL_ID;
		indexMap[i] = 0;
		signalMap[i] = 0;
		attackMap[i] = 0;
		mediumMap[i] = Constants::mediumInitial;
	}
	
	gpuUploadMaps();

}

size_t Simulation::getAgentAt(Position2i pos) {
	pos.wrapPositive(Constants::mapSize, Constants::mapSize);
	return indexMap[pos.y + pos.x * Constants::mapSize];
}

void Simulation::addAgentFromSpecie(SpecieID sid, Position2i pos) {
	if (agents.size() == Constants::totalMapSize) return;

	AgentResourceID id = getAgentID();
	SpecieID specieId = sid;

	registerNewSpecieMember(specieId);

	// Register and allocate specie and agent resources

	SIE_Manager.registerAgent(id);
	SG_Manager.registerAgent(id);
	AP_Manager.registerAgent(id);
	Position2f zerof = Position2f(0, 0);
	Agent newAgent = { specieId,id,pos,zerof,pos,0,Constants::initialFood,Constants::agentLifetime };

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

	Position2i pos;

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
	Position2f zerof = Position2f(0, 0);
	Agent newAgent = { specieId,id,pos,zerof,pos,0,Constants::initialFood,Constants::agentLifetime };

	agents.push_back(newAgent); // Agent vec
	xPositions.push_back(newAgent.pos.x); // x vec
	yPositions.push_back(newAgent.pos.y); // y vec
	foodLevels.push_back(newAgent.food.toFloat());
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId; // specieMap
	indexMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = agents.size() - 1;

}

bool Simulation::addAgent(Agent parent) {

	Position2i pos = parent.lastPos;

	if (positionOccupied(pos)) {

		// Algorithm for random position selection around the parent agent

		bool l = !positionOccupied(parent.pos + Position2i::left);
		bool r = !positionOccupied(parent.pos + Position2i::right);
		bool d = !positionOccupied(parent.pos + Position2i::down);
		bool u = !positionOccupied(parent.pos + Position2i::up);

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

		Position2i deltas[4] = { Position2i::left, Position2i::right, Position2i::up, Position2i::down };

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

	Position2f zerof = Position2f(0, 0);
	Agent newAgent = { specieId,id,pos,zerof,pos,generation,Constants::initialFood,Constants::agentLifetime };

	spillFood(pos, Constants::multiplyEnergyCost - Constants::initialFood);

	agents.push_back(newAgent);
	xPositions.push_back(newAgent.pos.x);
	yPositions.push_back(newAgent.pos.y);
	foodLevels.push_back(newAgent.food.toFloat());
	specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;
	indexMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = agents.size() - 1;

	Rational medium = mediumMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize];
	mediumMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = { 0,1 };
	Position2i delta = newAgent.pos - pos;

	if (Constants::totalMapSize > agents.size()) {
	for (int j = 0;j < Constants::mapSize;j++) {
		for (int i = 1;i < Constants::mapSize; i++) {

			Position2i now = delta * i + newAgent.pos + Position2i(j,j);
			now.wrapPositive(Constants::mapSize, Constants::mapSize);
			if (specieMap[now.y + now.x * Constants::mapSize] == NULL_ID) {
				setMediumAt(now, getMediumAt(now) + medium);
				return true;
			}

		}
	}
	}


	return true;
}

void Simulation::removeAgent(size_t index) {
	Agent removed = agents[index];

	spillFood(removed.pos, removed.food);

	// Non index based maps can be updated immediatly
	signalMap[removed.pos.y + removed.pos.x * Constants::mapSize] = 0;		 // These two make the agent invisible
	specieMap[removed.pos.y + removed.pos.x * Constants::mapSize] = NULL_ID; // to the simulation
	attackMap[removed.pos.y + removed.pos.x * Constants::mapSize] = 0;

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

	Position2i pos = agents[index].pos;
	indexMap[pos.y + pos.x * Constants::mapSize] = index;
}

SpecieID Simulation::newSpiecie(size_t parent) {
	SpecieID id = getSpecieID();

	Tensor specieSignal = specieSignalAllocator.getTensor();

	specieConrelationMap[id] = parent;
	if (parent == NULL_ID) {

		specieColorPallete[id] = randomAgentColor();

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
		specieColorPallete[id] = mutateColor(specieColorPallete[parent]);

		SIE_Manager.registerSpecie(
			parent,
			id,
			Constants::SIE_MutationDetails.mutationProbability,
			-Constants::SIE_MutationDetails.mutationAmplitude,
			Constants::SIE_MutationDetails.mutationAmplitude,
			Constants::SIE_MutationDetails.zeroMutationProbability
		);
		
		//SG_Manager.registerSpecie(
		//	parent,
		//	id,
		//	Constants::SG_MutationDetails.mutationProbability,
		//	-Constants::SG_MutationDetails.mutationAmplitude,
		//	Constants::SG_MutationDetails.mutationAmplitude,
		//	Constants::SG_MutationDetails.zeroMutationProbability
		//);

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

	specieIDs.push_back(id);
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
		specieIDs.erase(std::find(specieIDs.begin(), specieIDs.end(), specie));
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

bool Simulation::positionOccupied(Position2i pos) {
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

void Simulation::setAgentPos(size_t index, Position2i newPos) {
	specieMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize] = 0;
	signalMap[newPos.y + newPos.x * Constants::mapSize] = signalMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize];
	attackMap[newPos.y + newPos.x * Constants::mapSize] = attackMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize];
	signalMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize] = 0;
	attackMap[agents[index].pos.y + agents[index].pos.x * Constants::mapSize] = 0;

	agents[index].lastPos = agents[index].pos;
	agents[index].pos = newPos;

	xPositions[index] = newPos.x;
	yPositions[index] = newPos.y;

	specieMap[newPos.y + newPos.x * Constants::mapSize] = agents[index].specieId;
	indexMap[newPos.y + newPos.x * Constants::mapSize] = index;
}

void Simulation::moveAgent(size_t index, Position2i delta) {
	// Todo: Move signal
	Position2i to = agents[index].pos + delta;

	to.wrapPositive(Constants::mapSize, Constants::mapSize);

	if (!positionOccupied(to)) {
		//if (Random::runProbability(0.2f)) {
		//}
		Position2i front = to + delta;

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

void Simulation::spillFood(Position2i pos, Rational amount) { 
	if (amount.negative()) __debugbreak();
	setFoodAt(pos, getFoodAt(pos) + amount);

}

void Simulation::eat(size_t index,Rational amount) {
	for (int i = 0; i < 4; i++) {
		Position2i deltas[4] = { Position2i::left, Position2i::right, Position2i::up, Position2i::down };
		Position2i pos = agents[index].pos + deltas[i];
		pos.wrapPositive(Constants::mapSize, Constants::mapSize);
		Rational oneOverFour = { 1,4 };
		amount = amount * oneOverFour;
		if (getFoodAt(pos) >= amount && agents[index].food + amount >= Rational()) {
			addToAgentFood(index, amount);
			setFoodAt(pos, getFoodAt(pos) - amount);
		}

		if (getFoodAt(pos) < amount) {
			addToAgentFood(index, getFoodAt(pos));
			setFoodAt(pos, { 0,1 });
		}

		if (agents[index].food + amount < Rational()){
			setFoodAt(pos, getFoodAt(pos) + agents[index].food);
			addToAgentFood(index, -agents[index].food);
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

void Simulation::transfer(size_t index,Rational amount) {

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

	Position2i deltas[4] = { Position2i::left, Position2i::right, Position2i::up, Position2i::down };

	Rational oneOverFour = { 1,4 };
	amount = amount * oneOverFour;

	for (int i = 0; i < 4; i++) {
		Position2i pos = agents[index].pos + deltas[i];
		pos.wrapPositive(Constants::mapSize, Constants::mapSize);
		if (positionOccupied(pos)) {
			size_t target = getAgentAt(pos);

			if (amount <= agents[target].food && amount + agents[index].food >= Rational()) {
				addToAgentFood(index, amount);
				addToAgentFood(target, -amount);
			}

			if (amount > agents[target].food) {
				addToAgentFood(index, agents[target].food);
				addToAgentFood(target, -agents[target].food);
			}

			if (amount + agents[index].food < Rational()) {
				addToAgentFood(target, agents[index].food);
				addToAgentFood(index, -agents[index].food);
				return;
			}
		}

	}
}

void Simulation::share(size_t index,Rational amount) {

	Rational oneOverFour = { 1,4 };
	Rational sharable =amount * oneOverFour;

	Position2i pos = agents[index].pos;

	Position2i deltas[4] = { Position2i::left, Position2i::right, Position2i::up, Position2i::down };

	for (int i = 0; i < 4; i++) {
		Position2i relative = pos + deltas[i];
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
	//SG_Manager.compile(&agents[from], to - from);
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

	//gpuSync();

	//profiler.start(SG_PREDICT_ROUTINE);
	//Tensor generatedSignals = SG_Manager.predict(slicedAPSG);
	//profiler.end(SG_PREDICT_ROUTINE);

	gpuSync();

	profiler.start(AP_PREDICT_ROUTINE);
	Tensor decisions = AP_Manager.predict(slicedAPSG);
	profiler.end(AP_PREDICT_ROUTINE);

	gpuSync();

	decisions.getValue(&decisionOutput[from*6]);
	//generatedSignals.getValue(&generatedSignalsOutput[from]);

}

void Simulation::update() {

	profiler.reset();

	if (paused && !step) return;
	if (step) step = !step;

	profiler.start(UPDATE_DELTA);

	aa++;

	gpuSync();

	gpuUploadMaps();

	size_t remaining = agents.size();
	size_t current = 0;

	for (int i = 0; i < 6; i++) actionTracker[i] = 0;

	while(agents.size() * 6 >= decisionOutput.size()) decisionOutput.resize(2 * decisionOutput.size());
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
			Position2i pos(x, y);
			//spillFood(pos, { 1,100 });
			std::random_shuffle(deltas.begin(), deltas.end());
			for (int i = 0; i < 4; i++) {
				Position2i dp = (pos + deltas[i]);
				if (positionOccupied(dp)) continue;
				dp.wrapPositive(Constants::mapSize, Constants::mapSize);
				if (getFoodAt(dp) < getFoodAt(pos)) {
					Rational f = getFoodAt(pos) - getFoodAt(dp);
					if (f.b < 100) f.multiply(100);
					f.a /= 2;
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
		Position2i pos = agents[i].pos;

		for (int j = 0; j < 6; j++) actionTracker[j] += decisionOutput[i *6 + j];

		if (decisionOutput[i * 6 + MULTIPLY] > 0) {
			if (agents[i].food > Constants::multiplyEnergyCost) {
				if (addAgent(agents[i])) addToAgentFood(i, -Constants::multiplyEnergyCost);
			}
			else {
				//spillFood(agent.pos, agent.food);
				//addToAgentFood(i, -agent.food);
			}
			//continue;
		}

		Rational attackGain = Constants::attackEnergyGain;
		if (attackGain.b < 100) attackGain.multiply(100);
		attackGain.a = attackGain.a * decisionOutput[i * 6 + TRANSFER];
		attackMap[pos.y + pos.x * Constants::mapSize] = decisionOutput[i * 6 + TRANSFER];
		transfer(i, attackGain);

	    signalMap[pos.y + pos.x * Constants::mapSize] = decisionOutput[i * 6 + SIGNAL];
		
		Rational eatAmount = Constants::eatAmount;
		if (eatAmount.b < 100) eatAmount.multiply(100);
		eatAmount.a = eatAmount.a * decisionOutput[i * 6 + EAT];

		eat(i, eatAmount);
		Position2f posf = Position2f(decisionOutput[i * 6 + MOVE_X], decisionOutput[i * 6 + MOVE_Y]);
		Position2f r = agents[i].currentPos;
		agents[i].currentPos = agents[i].currentPos + posf;
		if (abs(agents[i].currentPos.x) >= 1 || abs(agents[i].currentPos.y) >= 1) {
			Position2i delta = agents[i].currentPos.to2i();
			moveAgent(i, delta);
			agents[i].currentPos = agents[i].currentPos - delta.to2f();
		}
	}

	//Rational r = agents[20].food;
	profiler.end(DECISION_PROCESS_ROUTINE);
	
	std::vector<size_t> toRemove;

	for (size_t i = agents.size() - 1; i < agents.size(); i--) {
		if (agents[i].food <= Rational() || agents[i].food > Constants::maximumFood || --agents[i].lifetime <= 0) toRemove.push_back(i);
	}

	for (size_t i = 0; i < toRemove.size(); i++) {
		removeAgent(toRemove[i]);
	}

	profiler.end(UPDATE_DELTA);

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

float* Simulation::getAttackMap() {
	return attackMap;
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
		<< "Update delta: " << profiler.get(UPDATE_DELTA) << " ms\n"
		<< "Total simulation food: " << getTotalFood() << '\n'
		<< "Total simulation medium: " << getTotalMedium() << '\n'
		<< "Agent energy: " <<ae << '\n'
		<< "Energy per agent: " << ae/agents.size() << '\n'
		<< aa<<'\n'
		<< "Actions: \n";

	for (int i = 0; i < 6; i++) std::cout << (float)actionTracker[i] / agents.size() << '\n';
 
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

Rational Simulation::getFoodAt(Position2i pos) {
	return rationalMapFood[pos.y + pos.x * Constants::mapSize];
}

void Simulation::setFoodAt(Position2i pos, Rational r) {

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

Rational Simulation::getMediumAt(Position2i pos) {
	return mediumMap[pos.y + pos.x * Constants::mapSize];
}

void Simulation::setMediumAt(Position2i pos,Rational value) {
	mediumMap[pos.y + pos.x * Constants::mapSize] = value;
}

float Simulation::getTotalMedium() {
	Rational total = { 0,1 };

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		total += mediumMap[i];
	}

	return total.toFloat();
}

Position2i Simulation::randomUnoccupiedPosition() {
	Position2i pos;

	do {
		pos = randomPositionGenerator.next();
	} while (positionOccupied(pos));

	return pos;
}

void Simulation::togglePause() {
	paused = !paused;
}

void Simulation::saveSimulationState(const char* path) {
	std::cout << "Saving simulation state...\n";

	HighFive::File file(path, HighFive::File::Overwrite);

	file.createDataSet("sqgen/time", aa);

	std::vector<long long> numerators(Constants::totalMapSize);
	std::vector<long long> denominators(Constants::totalMapSize);

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		numerators[i] = rationalMapFood[i].numerator();
		denominators[i] = rationalMapFood[i].denominator();
	}

	file.createDataSet("sqgen/maps/energy/numerators", numerators);
	file.createDataSet("sqgen/maps/energy/denominators", denominators);

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		numerators[i] = mediumMap[i].numerator();
		denominators[i] = mediumMap[i].denominator();
	}

	file.createDataSet("sqgen/maps/medium/numerators", numerators);
	file.createDataSet("sqgen/maps/medium/denominators", denominators);

	std::vector<float> signalMapVec(Constants::totalMapSize);

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		signalMapVec[i] = signalMap[i];
	}

	file.createDataSet("sqgen/maps/signal", signalMapVec);

	file.createDataSet("sqgen/agents/count", agents.size());
	file.createDataSet("sqgen/agents/spiecie_counter", specieCounter);

	std::vector<short> posX(agents.size());
	std::vector<short> posY(agents.size());

	for (size_t i = 0; i < agents.size(); i++) {
		posX[i] = agents[i].pos.x;
		posY[i] = agents[i].pos.y;
	}

	file.createDataSet("sqgen/agents/x", posX);
	file.createDataSet("sqgen/agents/y", posY);

	std::vector<SpecieID> spiecies(agents.size());

	for (size_t i = 0; i < agents.size(); i++) {
		spiecies[i] = agents[i].specieId;
	}

	file.createDataSet("sqgen/agents/spiecie_ids", spiecies);

	std::vector<short> lastX(agents.size());
	std::vector<short> lastY(agents.size());

	for (size_t i = 0; i < agents.size(); i++) {
		lastX[i] = agents[i].lastPos.x;
		lastY[i] = agents[i].lastPos.y;
	}

	file.createDataSet("sqgen/agents/last_x", lastX);
	file.createDataSet("sqgen/agents/last_y", lastY);

	std::vector<float> progX(agents.size());
	std::vector<float> progY(agents.size());

	for (size_t i = 0; i < agents.size(); i++) {
		progX[i] = agents[i].currentPos.x;
		progY[i] = agents[i].currentPos.y;
	}

	file.createDataSet("sqgen/agents/prog_x", progX);
	file.createDataSet("sqgen/agents/prog_y", progY);

	std::vector<size_t> generations(agents.size());
	std::vector<size_t> lifetimes(agents.size());

	std::vector<long long> foodN(agents.size());
	std::vector<long long> foodD(agents.size());

	for (size_t i = 0; i < agents.size(); i++) {
		generations[i] = agents[i].generation;
		lifetimes[i] = agents[i].lifetime;
		foodN[i] = agents[i].food.numerator();
		foodD[i] = agents[i].food.denominator();
	}


	file.createDataSet("sqgen/agents/generations", generations);
	file.createDataSet("sqgen/agents/lifetimes", lifetimes);
	file.createDataSet("sqgen/agents/energy/numerator", foodN);
	file.createDataSet("sqgen/agents/energy/denominator", foodD);

	file.createDataSet("sqgen/species/specie_count/", specieIDs.size());
	file.createDataSet("sqgen/species/specie_list/", specieIDs);

	std::string speciePath("sqgen/species/");

	for (size_t i = 0; i < specieIDs.size(); i++) {

		SpecieID id = specieIDs[i];

		std::string currentS = speciePath + std::to_string(i) + "/";

		Color c = specieColorPallete[id];

		std::vector<char> rgb{c.r,c.g,c.b};

		file.createDataSet(currentS + "color", rgb);

		std::vector<float> signal(Constants::spicieSignalCount);

		specieSignalDict[id].getValue(signal.data());

		file.createDataSet(currentS + "signal", signal);

		Tensor* varsSIE = SIE_Manager.agentModelVariables[id];

		for (size_t j = 0; j < SIE_Manager.paramCount; j++) {
			std::vector<float> p(varsSIE[j].size.size);
			varsSIE[j].getValue(p.data());
			file.createDataSet(currentS + "SIE/params/p" + std::to_string(j), p);
		}

		Tensor* varsAP = AP_Manager.agentModelVariables[id];

		for (size_t j = 0; j < AP_Manager.paramCount; j++) {
			std::vector<float> p(varsAP[j].size.size);
			varsAP[j].getValue(p.data());
			file.createDataSet(currentS + "AP/params/p" + std::to_string(j), p);
		}

	}
	std::cout << "Saved!\n";

}

void Simulation::loadSimulationState(const char* path) {

	std::cout << "Loading simulation state...\n";

	while (agents.size() != 0) removeAgent(0); // Dump all agent related data
	specieInstaceCounter.clear();

	HighFive::File file(path, HighFive::File::ReadOnly);

	file.getDataSet("sqgen/time").read(aa);

	std::vector<long long> numerators;
	std::vector<long long> denominators;

	file.getDataSet("sqgen/maps/energy/numerators").read(numerators);
	file.getDataSet("sqgen/maps/energy/denominators").read(denominators);

	for (size_t i = 0; i < Constants::totalMapSize; i++) {

		rationalMapFood[i] = { numerators[i] , denominators[i] };

	}

	file.getDataSet("sqgen/maps/medium/numerators").read(numerators);
	file.getDataSet("sqgen/maps/medium/denominators").read(denominators);

	for (size_t i = 0; i < Constants::totalMapSize; i++) {

		mediumMap[i] = { numerators[i] , denominators[i] };

	}

	std::vector<float> signalMapVec;

	file.getDataSet("sqgen/maps/signal").read(signalMapVec);

	for (size_t i = 0; i < Constants::totalMapSize; i++) {
		signalMap[i] = signalMapVec[i];
	}

	size_t agentCount;

	file.getDataSet("sqgen/agents/count").read(agentCount);
	file.getDataSet("sqgen/agents/spiecie_counter").read(specieCounter);

	std::vector<short> posX;
	std::vector<short> posY;
	std::vector<SpecieID> spiecies;
	std::vector<short> lastX;
	std::vector<short> lastY;
	std::vector<float> progX;
	std::vector<float> progY;
	std::vector<size_t> generations;
	std::vector<size_t> lifetimes;
	std::vector<long long> foodN;
	std::vector<long long> foodD;

	file.getDataSet("sqgen/agents/x").read(posX);
	file.getDataSet("sqgen/agents/y").read(posY);
	file.getDataSet("sqgen/agents/spiecie_ids").read(spiecies);
	file.getDataSet("sqgen/agents/last_x").read(lastX);
	file.getDataSet("sqgen/agents/last_y").read(lastY);
	file.getDataSet("sqgen/agents/prog_x").read(progX);
	file.getDataSet("sqgen/agents/prog_y").read(progY);
	file.getDataSet("sqgen/agents/generations").read(generations);
	file.getDataSet("sqgen/agents/lifetimes").read(lifetimes);
	file.getDataSet("sqgen/agents/energy/numerator").read(foodN);
	file.getDataSet("sqgen/agents/energy/denominator").read(foodD);

	size_t specieIDscount;

	file.getDataSet("sqgen/species/specie_count/").read(specieIDscount);
	file.getDataSet("sqgen/species/specie_list/").read(specieIDs);

	std::string speciePath("sqgen/species/");

	for (size_t i = 0; i < specieIDs.size(); i++) {

		SpecieID id = specieIDs[i];

		std::string currentS = speciePath + std::to_string(i) + "/";

		std::vector<char> rgb;

		file.getDataSet(currentS + "color").read(rgb);

		specieColorPallete[id] = Color(rgb[0], rgb[1], rgb[2]);

		std::vector<float> signal(Constants::spicieSignalCount);

		file.getDataSet(currentS + "signal").read(signal);

		Tensor sig(Size(2, 1, Constants::spicieSignalCount));
		sig.setValue(signal.data());

		specieSignalDict[id] = sig;

		std::vector<std::vector<float>> paramsSIE(SIE_Manager.paramCount);

		for (size_t j = 0; j < SIE_Manager.paramCount; j++) {
			file.getDataSet(currentS + "SIE/params/p" + std::to_string(j)).read(paramsSIE[j]);
		}

		SIE_Manager.loadSpiecie(paramsSIE.data(), id);


		std::vector<std::vector<float>> paramsAP(AP_Manager.paramCount);

		for (size_t j = 0; j < AP_Manager.paramCount; j++) {
			file.getDataSet(currentS + "AP/params/p" + std::to_string(j)).read(paramsAP[j]);
		}

		AP_Manager.loadSpiecie(paramsAP.data(), id);

	}

	for (size_t i = 0; i < agentCount; i++) {
		AgentResourceID id = getAgentID();

		SIE_Manager.registerAgent(id);
		SG_Manager.registerAgent(id);
		AP_Manager.registerAgent(id);

		Position2i pos(posX[i], posY[i]);
		Position2f prog(progX[i], progY[i]);
		Position2i lastPos(lastX[i], lastY[i]);

		Agent newAgent = { spiecies[i],id,pos,prog,lastPos,generations[i],{foodN[i],foodD[i]},lifetimes[i]};

		agents.push_back(newAgent);
		xPositions.push_back(newAgent.pos.x);
		yPositions.push_back(newAgent.pos.y);
		foodLevels.push_back(newAgent.food.toFloat());
		specieMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = newAgent.specieId;
		indexMap[newAgent.pos.y + newAgent.pos.x * Constants::mapSize] = agents.size() - 1;
		if (!specieInstaceCounter.count(spiecies[i])) specieInstaceCounter[spiecies[i]] = 0;
		registerNewSpecieMember(spiecies[i]);
	}
	std::cout << "Loaded\n";

}