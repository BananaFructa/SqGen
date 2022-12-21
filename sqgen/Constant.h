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
		10,		// Init Inputs
		20,		// Init Hidden
		1.0f	// Init Amplitude
	};

	const NetworkInitDetails SG_InitDetails{
		9,		// Init Inputs
		5,		// Init Hidden
		1		// Init Amplitude
	};	

	const NetworkInitDetails SIE_InitDetails = {
		0,		// No use
		0,		// No use
		1.0f	// Init Ampltitude
	};

	// =================== SIMULATION CONSTANTS =====================

	const size_t mapSize = 200;
	const size_t totalMapSize = mapSize * mapSize;

	const int agentObserveRange = 2;
	const int agentObserveRangeTotal = (agentObserveRange * 2 + 1) * (agentObserveRange * 2 + 1);
	const float specieSignalMutationProb = 0;
	const float specieSignalMutatuionAmplitude = 0;

	const float agentMutationProbability = 1;

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

	const float initialMapFood = 1;

	const float maximumFood = 30;
	const float initialFood = 10;
	const float eatAmount = 0;
	const float moveEnergyCost = 0;
	const float multiplyEnergyCost = 0;
	const float attackEnergyCost = 0;
	const float attackEnergyGain = 0;

	const float shareEnergyTransfer = 0;
	const int shareRadius = 0;
}
