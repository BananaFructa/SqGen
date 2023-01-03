#pragma once

#include "Rational.hpp"

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
	constexpr size_t curandPoolSize = 10000;
	constexpr unsigned long seed = 780;

	// ========= NN INITIALIZATION AND INTERFACING CONSTANTS =======

	constexpr size_t spicieSignalCount = 3;
	constexpr float specieSignalAmplitude = 1;

	constexpr size_t visualLatentSize = 1;

	const NetworkInitDetails AP_InitDetails = {
		10,		// Init Inputs
		5,		// Init Hidden
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
		1	// Init Ampltitude
	};

	// =================== SIMULATION CONSTANTS =====================

	constexpr size_t mapSize = 200;
	constexpr size_t totalMapSize = mapSize * mapSize;

	constexpr int agentObserveRange = 2;
	constexpr int agentObserveRangeTotal = (agentObserveRange * 2 + 1) * (agentObserveRange * 2 + 1);
	const float specieSignalMutationProb = 0.1;
	const float specieSignalMutatuionAmplitude = 0.03;

	const float agentMutationProbability = 0.85;

	const NetworkMutationDetails AP_MutationDetails = {
		1.0,		// Non-zero mutation probability
		0.01,		// Zero mutation probability
		0.03		// Mutation Amplitude
	};

	const NetworkMutationDetails SG_MutationDetails = {
		1.0,		// Non-zero mutation probability
		0.3,		// Zero mutation probability
		0.03		// Mutation Amplitude	// Mutation Amplitude
	};

	const NetworkMutationDetails SIE_MutationDetails = {
		1.0,		// Non-zero mutation probability
		0.3,		// Zero mutation probability
		0.03	// Mutation Amplitude		// Mutation Amplitude
	};

	constexpr Rational initialMapFood = {  1, 1	};
	constexpr Rational maximumFood = { 10,1 }; // when dead the agent should spill
	const Rational initialFood = { 1,1 };
	const Rational eatAmount = { 1, 1 };
	const Rational moveEnergyCost = { 1, 2 }; // spilled
	const Rational multiplyEnergyCost = { 2, 1 }; // this energy should also be included in the spillage

	constexpr float FinitialMapFood = (float)initialMapFood.a / initialMapFood.b;
	constexpr float FmaximumFood = (float)maximumFood.a / maximumFood.b;

	const Rational attackEnergyCost = { 1,8 }; // make energy be spilled on the map
	const Rational attackEnergyGain = { 1,4 };

	const Rational shareEnergyTransfer = { 1,4 };
	const int shareRadius = 1;

	const int spillRange = 2;
	const Rational spillMap[(spillRange * 2 + 1) * (spillRange * 2 + 1)] = {
		{0,1},			{0,1},			{0,1},		{0,1},			{0,1},
		{0,1},			{1,10},			{1,10},		{1,10},			{0,1},
		{0,1},			{1,10},			{1,10} ,	{1,10},			{0,1},
		{0,1},			{1,10},			{1,10},		{1,10},			{0,1},
		{0,1},			{0,1},			{1,10},		{0,1},			{0,1}
	};

	const int startingAgentCount = 40000;
}
