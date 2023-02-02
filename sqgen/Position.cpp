#include "Position.hpp"

Position2i Position2i::left = Position2i(-1, 0);
Position2i Position2i::right = Position2i(1, 0);
Position2i Position2i::up = Position2i(0, -1);
Position2i Position2i::down = Position2i(0, 1);

Position2f Position2i::to2f() {
	return Position2f((float)x, (float)y);
}