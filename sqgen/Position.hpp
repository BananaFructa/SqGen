#pragma once
struct Position {

	static Position left;
	static Position right;
	static Position up;
	static Position down;

	size_t x;
	size_t y;

	const bool operator==(Position& b) {
		return this->x == b.x && this->y == b.y;
	}

	const Position operator+(Position& b) {
		return Position(this->x + b.x, this->y + b.y);
	}

	Position() {

	}

	Position(size_t x, size_t y) {
		this->x = x;
		this->y = y;
	}

	void wrapPositive(size_t xMax, size_t yMax) {
		if (x < 0) x += xMax;
		if (x >= xMax) x -= xMax;
		if (y < 0) y += yMax;
		if (y >= yMax) y -= yMax;
	}
};