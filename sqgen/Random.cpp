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

size_t Random::runProbabilityVector(float* vec, size_t size) {
    float rnd = ((float)rand() / (float)RAND_MAX);

    float sum = 0;

    for (size_t i = 0; i < size; i++) {
        if (vec[i] == 0) continue;
        if (vec[i] == 1) return i;
        sum += vec[i];
        if (sum >= rnd) return i;
    }

    return size - 1;
}
