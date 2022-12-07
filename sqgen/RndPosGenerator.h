#pragma once

#include <vector>
#include <algorithm>
#include "Position.h"

struct RndPosGenerator {

	size_t current = 0;
	std::vector<Position> shuffeledPositions;

	RndPosGenerator() {

	}

	RndPosGenerator(Position start,Position end) {
		for (size_t x = start.x; x < end.x; x++) {
			for (size_t y = start.y; y < end.y; y++) {
				shuffeledPositions.push_back(Position(x, y));
			}
		}
		std::random_shuffle(shuffeledPositions.begin(), shuffeledPositions.end());
	}

	Position next() {
		if (current == shuffeledPositions.size()) current = 0;
		return shuffeledPositions[current++];
	}

};