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

	constexpr size_t nnPoolSize = 100'000;
	constexpr size_t curandPoolSize = 1000;
	constexpr unsigned long seed = 177;

	// ========= NN INITIALIZATION AND INTERFACING CONSTANTS =======

	constexpr size_t spicieSignalCount = 10;
	constexpr float specieSignalAmplitude = 1;

	constexpr size_t visualLatentSize = 1;

	const NetworkInitDetails AP_InitDetails = {
		6,		// Init Inputs
		10,		// Init Hidden
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

	constexpr size_t mapSize = 400;
	constexpr size_t totalMapSize = mapSize * mapSize;

	constexpr int agentObserveRange = 2;
	constexpr int agentObserveRangeTotal = (agentObserveRange * 2 + 1) * (agentObserveRange * 2 + 1);
	constexpr float specieSignalMutationProb = 0.5;
	constexpr float specieSignalMutatuionAmplitude = 0.1;

	constexpr float agentMutationProbability = 0.1;

	const NetworkMutationDetails AP_MutationDetails = {
		0.15,		// Non-zero mutation probability
		0.001,		// Zero mutation probability
		0.1		// Mutation Amplitude
	};

	const NetworkMutationDetails SG_MutationDetails = {
		0.05,		// Non-zero mutation probability
		0.01,		// Zero mutation probability
		0.2		// Mutation Amplitude	// Mutation Amplitude
	};

	const NetworkMutationDetails SIE_MutationDetails = {
		0.1,		// Non-zero mutation probability
		0,		// Zero mutation probability
		0.1		// Mutation Amplitude		// Mutation Amplitude
	};

	constexpr float initialMapFood = 1;
	constexpr float maximumFood = 5; // when dead the agent should spill
	constexpr float initialFood = 1;
	constexpr float eatAmount = 0.1;
	constexpr float moveEnergyCost = 0.01; // spilled
	constexpr float multiplyEnergyCost = 2; // this energy should also be included in the spillage

	constexpr float attackEnergyCost = 0.1; // make energy be spilled on the map
	constexpr float attackEnergyGain = 0.5;

	constexpr float shareEnergyTransfer = 0.3;
	constexpr int shareRadius = 1;

	constexpr float foodIncrease = 0.001; // this needs to be removed
}
