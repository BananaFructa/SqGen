#pragma once

#include "Position.hpp"
#include "Rational.hpp"

#define NULL_ID 0

typedef size_t SpecieID;
typedef size_t AgentResourceID;

struct Agent {

	// The spieci identifier of the agent (used for param NN values)
	// Unique on the whole existance of the simulation instance
	SpecieID specieId;
	// Unique identifier for the specific agent used for resource managment
	AgentResourceID id = 0;
	// Pos
	Position pos;

	Position lastPos;

	size_t generation = 0;

	Rational food;

};