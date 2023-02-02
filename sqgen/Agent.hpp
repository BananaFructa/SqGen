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
	Position2i pos;
	Position2f currentPos = Position2f(0,0);

	Position2i lastPos;

	size_t generation = 0;

	Rational food;

	size_t lifetime;

};