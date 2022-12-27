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
	constexpr size_t curandPoolSize = 1000;
	constexpr unsigned long seed = 747;

	// ========= NN INITIALIZATION AND INTERFACING CONSTANTS =======

	constexpr size_t spicieSignalCount = 10;
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
		1.0f	// Init Ampltitude
	};

	// =================== SIMULATION CONSTANTS =====================

	constexpr size_t mapSize = 250;
	constexpr size_t totalMapSize = mapSize * mapSize;

	constexpr int agentObserveRange = 2;
	constexpr int agentObserveRangeTotal = (agentObserveRange * 2 + 1) * (agentObserveRange * 2 + 1);
	constexpr float specieSignalMutationProb = 0.1;
	constexpr float specieSignalMutatuionAmplitude = 0.1;

	constexpr float agentMutationProbability = 0.45;

	const NetworkMutationDetails AP_MutationDetails = {
		0.34,		// Non-zero mutation probability
		0.01,		// Zero mutation probability
		0.5		// Mutation Amplitude
	};

	const NetworkMutationDetails SG_MutationDetails = {
		0.34,		// Non-zero mutation probability
		0.01,		// Zero mutation probability
		0.5		// Mutation Amplitude	// Mutation Amplitude
	};

	const NetworkMutationDetails SIE_MutationDetails = {
		0.34,		// Non-zero mutation probability
		0,		// Zero mutation probability
		0.3		// Mutation Amplitude		// Mutation Amplitude
	};

	constexpr Rational initialMapFood = {  1,2	};
	constexpr Rational maximumFood = { 10,1 }; // when dead the agent should spill
	constexpr Rational initialFood = { 1,1 };
	constexpr Rational eatAmount = { 1, 2 };
	constexpr Rational moveEnergyCost = { 1, 20 }; // spilled
	constexpr Rational multiplyEnergyCost = { 2, 1 }; // this energy should also be included in the spillage

	constexpr float FinitialMapFood = (float)initialMapFood.a / initialMapFood.b;
	constexpr float FmaximumFood = (float)maximumFood.a / maximumFood.b;

	constexpr Rational attackEnergyCost = { 1,10 }; // make energy be spilled on the map
	constexpr Rational attackEnergyGain = { 1,2 };

	constexpr Rational shareEnergyTransfer = { 1,2 };
	constexpr int shareRadius = 1;

	constexpr int spillRange = 2;
	constexpr Rational spillMap[(spillRange * 2 + 1) * (spillRange * 2 + 1)] = {
		{0,1},			{0,1},			{0,1},		{0,1},			{0,1},
		{0,1},			{1,10},			{1,10},		{1,10},			{0,1},
		{0,1},			{1,10},			{1,10} ,	{1,10},			{0,1},
		{0,1},			{1,10},			{1,10},		{1,10},			{0,1},
		{0,1},			{0,1},			{1,10},		{0,1},			{0,1}
	};

	constexpr int startingAgentCount = 10000;
}
