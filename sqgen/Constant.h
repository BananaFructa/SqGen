#pragma once
namespace Constants {

	// ========== MEMORY AND PROCESS MANAGMENT CONSTANTS ===========

	const size_t nnPoolSize = 100'000'00;
	const size_t curandPoolSize = 1000;
	const unsigned long seed = 123;

	// ========= NN INITIALIZATION AND INTERFACING CONSTANTS =======

	const size_t spicieSignalCount = 10;
	const size_t visualLatentSize = 1;

	const size_t AP_InitializedInputs = 3;
	const size_t AP_InitializedHidden = 2;

	const float AP_InitAmplitude = 1;
	const float SIE_InitAmplitude = 1;

	// =================== SIMULATION CONSTANTS =====================

	const size_t mapSize = 50;
	const size_t totalMapSize = mapSize * mapSize;

	const size_t agentObserveRange = 2;

	const float agentMutationProbability = 0;

	const float AP_MutationProb = 0;
	const float SIE_MutationProb = 0;
	const float AP_ZeroMutationProb = 0;
	const float SIE_ZeroMutationProb = 0;
		
	const float AP_MutationAmplitude = 0;
	const float SIE_MutatuinAmplitude = 0;
}