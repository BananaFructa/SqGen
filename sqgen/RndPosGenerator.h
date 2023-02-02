#pragma once

#include <vector>
#include <algorithm>
#include "Position.hpp"

struct RndPosGenerator {

	size_t current = 0;
	std::vector<Position2i> shuffeledPositions;

	RndPosGenerator() {

	}

	RndPosGenerator(Position2i start,Position2i end) {
		for (size_t x = start.x; x < end.x; x++) {
			for (size_t y = start.y; y < end.y; y++) {
				shuffeledPositions.push_back(Position2i(x, y));
			}
		}
		std::random_shuffle(shuffeledPositions.begin(), shuffeledPositions.end());
	}

	Position2i next() {
		if (current == shuffeledPositions.size()) current = 0;
		return shuffeledPositions[current++];
	}

};