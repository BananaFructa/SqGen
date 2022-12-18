#pragma once
struct Position {

	static Position left;
	static Position right;
	static Position up;
	static Position down;

	int x;
	int y;

	const bool operator==(Position& b) {
		return this->x == b.x && this->y == b.y;
	}

	const Position operator+(Position& b) {
		return Position(this->x + b.x, this->y + b.y);
	}

	const Position operator*(int s) {
		return Position(this->x * s, this->y * s);
	}

	Position() {

	}

	Position(int x, int y) {
		this->x = x;
		this->y = y;
	}

	void wrapPositive(int xMax, int yMax) {
		if (x < 0) x += xMax;
		if (x >= xMax) x -= xMax;
		if (y < 0) y += yMax;
		if (y >= yMax) y -= yMax;
	}
};