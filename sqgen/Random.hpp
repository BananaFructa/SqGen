#pragma once
namespace Random {

	void setSeed(unsigned int seed);
	bool runProbability(float prob);
	int randomInt();
	float randomFloat();
	size_t runProbabilityVector(float* vec, size_t size);

}