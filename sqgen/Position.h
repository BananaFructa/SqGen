#pragma once
struct Position {

	size_t x;
	size_t y;

	const bool operator==(Position& b) {
		return this->x == b.x && this->y == b.y;
	}

	Position() {

	}

	Position(size_t x, size_t y) {
		this->x = x;
		this->y = y;
	}
};