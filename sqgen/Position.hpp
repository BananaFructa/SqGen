#pragma once
struct Position2f;
struct Position2i {

	static Position2i left;
	static Position2i right;
	static Position2i up;
	static Position2i down;

	int x;
	int y;

	const bool operator==(Position2i b) {
		return this->x == b.x && this->y == b.y;
	}

	const Position2i operator+(Position2i b) const {
		return Position2i(this->x + b.x, this->y + b.y);
	}

	const Position2i operator*(int s) const {
		return Position2i(this->x * s, this->y * s);
	}

	const Position2i operator-() const {
		return Position2i(-x, -y);
	}

	const Position2i operator-(Position2i b) const {
		return *this + (-b);
	}

	Position2i() {

	}

	Position2i(int x, int y) {
		this->x = x;
		this->y = y;
	}

	void wrapPositive(int xMax, int yMax) {
		x = (x + xMax) % xMax;
		y = (y + yMax) % yMax;
	}

	Position2f to2f();
};

struct Position2f {

	static Position2f left;
	static Position2f right;
	static Position2f up;
	static Position2f down;

	float x;
	float y;

	const bool operator==(Position2f b) {
		return this->x == b.x && this->y == b.y;
	}

	const Position2f operator+(Position2f b) const {
		return Position2f(this->x + b.x, this->y + b.y);
	}

	const Position2f operator*(int s) const {
		return Position2f(this->x * s, this->y * s);
	}

	const Position2f operator-() const {
		return Position2f(-x, -y);
	}

	const Position2f operator-(Position2f b) const {
		return *this + (-b);
	}

	Position2f() {

	}

	Position2f(float x, float y) {
		this->x = x;
		this->y = y;
	}

	Position2i to2i() {
		return Position2i((int)x, (int)y);
	}
};