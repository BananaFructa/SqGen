#pragma once
struct Position {

	static Position left;
	static Position right;
	static Position up;
	static Position down;

	int x;
	int y;

	const bool operator==(Position b) {
		return this->x == b.x && this->y == b.y;
	}

	const Position operator+(Position b) const {
		return Position(this->x + b.x, this->y + b.y);
	}

	const Position operator*(int s) const {
		return Position(this->x * s, this->y * s);
	}

	const Position operator-() const {
		return Position(-x, -y);
	}

	const Position operator-(Position b) const {
		return *this + (-b);
	}

	Position() {

	}

	Position(int x, int y) {
		this->x = x;
		this->y = y;
	}

	void wrapPositive(int xMax, int yMax) {
		x = (x + xMax) % xMax;
		y = (y + yMax) % yMax;
	}
};