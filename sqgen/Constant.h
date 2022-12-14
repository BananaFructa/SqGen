#pragma once

struct NetworkInitDetails {
public:
	size_t initializedInputs;
	size_t initializedHidden;
	float initAmplitude;
};

struct NetworkMutationDetails {
public:
	float mutationProbability;
	float zeroMutationProbability;
	float mutationAmplitude;
};

namespace Constants {

	// ========== MEMORY AND PROCESS MANAGMENT CONSTANTS ===========

	const size_t nnPoolSize = 100'000;
	const size_t curandPoolSize = 1000;
	const unsigned long seed = 126;

	// ========= NN INITIALIZATION AND INTERFACING CONSTANTS =======

	const size_t spicieSignalCount = 10;
	const float specieSignalAmplitude = 1;

	const size_t visualLatentSize = 1;

	const NetworkInitDetails AP_InitDetails = {
		3,		// Init Inputs
		2,		// Init Hidden
		1.0f	// Init Amplitude
	};

	const NetworkInitDetails SG_InitDetails{
		0,		// Init Inputs
		0,		// Init Hidden
		1		// Init Amplitude
	};	

	const NetworkInitDetails SIE_InitDetails = {
		0,		// No use
		0,		// No use
		1.0f	// Init Ampltitude
	};

	// =================== SIMULATION CONSTANTS =====================

	const size_t mapSize = 50;
	const size_t totalMapSize = mapSize * mapSize;

	const size_t agentObserveRange = 2;
	const size_t agentObserveRangeTotal = (agentObserveRange * 2 + 1) * (agentObserveRange * 2 + 1);
	const float specieSignalMutationProb = 0;
	const float specieSignalMutatuionAmplitude = 0;

	const float agentMutationProbability = 0;

	const NetworkMutationDetails AP_MutationDetails = {
		0,		// Non-zero mutation probability
		0,		// Zero mutation probability
		0		// Mutation Amplitude
	};

	const NetworkMutationDetails SG_MutationDetails = {
		0,		// Non-zero mutation probability
		0,		// Zero mutation probability
		0		// Mutation Amplitude
	};

	const NetworkMutationDetails SIE_MutationDetails = {
		0,		// Non-zero mutation probability
		0,		// Zero mutation probability
		0		// Mutation Amplitude
	};

	const float initialFood = 0;
	const float moveEnergyCost = 0;
	const float multiplyEnergyCost = 0;
	const float attackEnergyCost = 0;
	const float attackEnergyGain = 0;
}