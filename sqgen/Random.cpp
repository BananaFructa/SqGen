#include "Random.hpp"

#include <random>

void Random::setSeed(unsigned int seed) {
    srand(seed);
}

bool Random::runProbability(float prob) {
    return prob > (float)rand()/(float)RAND_MAX;
}

int Random::randomInt() {
    return rand();
}

float Random::randomFloat() {
    return ((float)rand() / (float)RAND_MAX) * 2 - 1;
}
