#pragma once

#include "Position.h"

#define NULL_PARENT 0

typedef size_t SpecieID;
typedef size_t AgentID;

struct Agent {

	// The spieci identifier of the agent (used for param NN values)
	// Unique on the whole existance of the simulation instance
	SpecieID specieId;
	// Unique identifier for the specific agent (used for state NN values)
	AgentID id;
	// Pos
	Position pos;

	size_t generation = 0;

};