#pragma once

#include <map>
#include <chrono>

class Profiler {
private:

	std::map<size_t, std::chrono::steady_clock::time_point> begin;
	std::map<size_t, double> elapsed;

	std::chrono::high_resolution_clock clock;
public:

	void start(size_t id);
	void end(size_t id);
	void reset();
	double get(size_t id);

};

