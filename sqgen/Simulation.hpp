#pragma once

#include <vector>
#include <map>
#include <mutex>
#include <chrono>

#include "Random.hpp"
#include "RndPosGenerator.h"
#include "Constant.h"
#include "Agent.hpp"
#include "Position.hpp"
#include "NNAgentModelManager.hpp"
#include "../ha_models/ReferenceMappedTensor.hpp"
#include "../ha_models/TensorMemAllocator.hpp"
#include "../ha_models/Array.hpp"
#include "Profiler.hpp"

#define EAT 0
#define MULTIPLY 1
#define UP 2
#define DOWN 3
#define RIGHT 4
#define LEFT 5
#define ATTACK 6
#define SHARE 7
#define SIGNAL 8

/*
	VIEW RANGE:

	# - The agent
	L/R/U/D = Left/Right/Up/Down

	For a view range of 2

	+---+---+---+---+---+
	|L U|L U| U |U R|U R|
	+---+---+---+---+---+
	|L U|L U| U |U R|U R|
	+---+---+---+---+---+
	| L | L | # | R | R |
	+---+---+---+---+---+
	|L D|L D| D |D R|D R|
	+---+---+---+---+---+
	|L D|L D| D |D R|D R|
	+---+---+---+---+---+


 */

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
												 +-------------------+					+--------------+
										  +----> |		  SG		 |---------X------> |			   |
				+-------------------+	  |	+--> +-------------------+		   |	    |			   |
		   +--> |		SIE			|-----|	|								   |		|	  SIM	   |
		   |	+-------------------+	  |	|	 +-------------------+		   |		|			   |
		   |							  +-|--> |		  AP		 |---------+------> |			   |
		   |								+--> +-------------------+					+--------------+
		   |								|												    |
		   |________________________________|___________________________________________________|
*/

struct Simulation {
public:
//  =======================================================================

	RndPosGenerator randomPositionGenerator;

//  =======================================================================

	CurandManager curandManager = CurandManager(Constants::curandPoolSize, Constants::seed);

	// Specie Information Encoder                         V each agent needs to run their SIE 4 times in 4 directions
	NNModel SIE_Network = NNModel(Constants::nnPoolSize * 4);
	NNAgentModelManager SIE_Manager;

	NNModel SG_Network = NNModel(Constants::nnPoolSize);
	NNAgentModelManager SG_Manager;

	// Action Processing
	NNModel AP_Netowrk = NNModel(Constants::nnPoolSize);
	NNAgentModelManager AP_Manager;

	void buildSIE(NNModel& model);
	void buildSG(NNModel& model);
	void buildAP(NNModel& model);

	// AP & SG gpu netowrk input             V 10 inputs             
	Tensor APSG_InputPool = Tensor(Size(3, 1, 10, Constants::nnPoolSize));

	// LEFT - RIGHT - UP - DOWN												one for each direction V
	Tensor SIE_InputPool = Tensor(Size(3, 1, Constants::spicieSignalCount * 2, Constants::nnPoolSize * 4));

	std::vector<float> generatedSignalsOutput;
	std::vector<float> decisionOutput;

//  =======================================================================

	std::vector<Agent> agents;
	
	// Linearly memeory stored positions for gpu upload
	// Used for input compiling
	std::vector<int> xPositions;
	std::vector<int> yPositions;
	Array<int> gpuXPositions = Array<int>(Constants::nnPoolSize);
	Array<int> gpuYPositions = Array<int>(Constants::nnPoolSize);

	std::vector<float> foodLevels;
	Array<float> gpuFoodLevels = Array<float>(Constants::nnPoolSize);


//  =======================================================================

	// ======
	// Used to avoid kernel bracnhes holds
	// Holds the directions to which the signal from the relative position should be attributed 
	// Structure of a short from the array 
	// | rest unused | 1 bit use second | 1 bit use first | 2 bits to indicate the second direction | 2 bits to indicate the direction |
	Array<short> logicMapObserveRange = Array<short>(Constants::agentObserveRangeTotal);
	Array<float> logicMapDistanceRange = Array<float>(Constants::agentObserveRangeTotal);
	// ======

	Rational* mediumMap = new Rational[Constants::totalMapSize];

	// Matrix which contain the food values of each tile
	float* foodMap = new float[Constants::totalMapSize];
	Rational* rationalMapFood = new Rational[Constants::totalMapSize];
	// GPU side matrix of the above
	Tensor gpuFoodMap = Tensor(Size(2, Constants::mapSize, Constants::mapSize));

	// Matrix which contains the signal outputs of the agents
	float* signalMap = new float[Constants::totalMapSize];
	// GPU side matrix of the above
	Tensor gpuSignalMap = Tensor(Size(2, Constants::mapSize, Constants::mapSize));

	// Matrix which contains the specie id of each agent
	SpecieID* specieMap = new SpecieID[Constants::totalMapSize];
	size_t* indexMap = new size_t[Constants::totalMapSize];

	// Matrix which contains a reference to the specie signal set of the agents at each position on the map
	ReferenceMappedTensor gpuSpecieSignalMap = ReferenceMappedTensor(Size(2, Constants::spicieSignalCount, Constants::totalMapSize));

	// Dictionary of specie ids and their signals
	std::map<SpecieID, Tensor> specieSignalDict;

	std::map<SpecieID, SpecieID> specieConrelationMap;

	// Allocator for specie signals
	TensorMemAllocator<Tensor> specieSignalAllocator = TensorMemAllocator<Tensor>(Size(1, Constants::spicieSignalCount));

//  =======================================================================

	SpecieID specieCounter = 1;
	AgentResourceID lastAgentID = 1;
	std::vector<AgentResourceID> avalabileAgentIDs;
	std::map<SpecieID, size_t> specieInstaceCounter;

//  =======================================================================

	Position dirs[4] = { Position::left, Position::right, Position::up, Position::down };

	std::vector<Position> deltas{ Position::left, Position::right, Position::up, Position::down };

	Profiler profiler;

	bool paused = true;
	bool step = false;

	size_t actionTracker[9] = {0};

	void eat(size_t index);
	void attack(size_t index);
	void share(size_t index);
	void addToAgentFood(size_t index, Rational food);
	void setAgentPos(size_t index, Position newPos);
	void moveAgent(size_t index, Position delta);
	bool positionOccupied(Position pos);
	void spillFood(Position pos, Rational amount);

public:

	Simulation();

	void gpuUploadMaps();

	size_t getAgentAt(Position pos);

	void addAgentFromSpecie(SpecieID id, Position pos);
	void addNewAgent();
	bool addAgent(Agent parent);
	void removeAgent(size_t index);

	SpecieID loadSpecie(const char* path);
	SpecieID newSpiecie(size_t parent);
	void registerNewSpecieMember(SpecieID specie);
	void eraseSpecieMember(SpecieID specie);
	SpecieID getParentSpecie(SpecieID child);

	AgentResourceID getAgentID();
	SpecieID getSpecieID();

	void pipelineToAPSG(size_t from, size_t to);
	void runAPSGAndProcessDecisions(size_t from, size_t to);
	void update();

	float* getFoodMap();
	float* getSignalMap();
	SpecieID* getSpecieMap();

	std::vector<Agent>& getAgents();
	std::map<SpecieID, Tensor>& getSignalDict();

	void printProfilerInfo();

	void togglePause();

	float getTotalFood();

	Rational getFoodAt(Position pos);
	void setFoodAt(Position pos, Rational r);
	void restartFoodMap();
	float getAgenentEnergy();

	Rational getMediumAt(Position pos);
	void setMediumAt(Position pos,Rational value);
	float getTotalMedium();

	Position randomUnoccupiedPosition();
};