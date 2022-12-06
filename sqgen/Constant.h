#pragma once
namespace Constants {

	// Pool allocation size of neural netowrks
	const size_t nnPoolSize = 100'000;

	// Pool allocation size of CUDA random number generator states
	const size_t curandPoolSize = 1000;

	// The seed of the CUDA number generators
	const unsigned long curandSeed = 123;

	// The length of the map (It is always a square)
	const size_t mapSize = 50;

	const size_t totalMapSize = mapSize * mapSize;

	const size_t spicieSignalCount = 10;

}