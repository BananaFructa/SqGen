#include "Profiler.hpp"

void Profiler::start(size_t id) {

	begin[id] = clock.now();

}

void Profiler::end(size_t id) {
	double duration = std::chrono::duration<double, std::milli>(std::chrono::duration_cast<std::chrono::milliseconds>(clock.now() - begin[id])).count();
	if (elapsed.count(id)) elapsed[id] += duration;
	else elapsed[id] = duration;
}

void Profiler::reset() {
	elapsed.clear();
	begin.clear();
}

double Profiler::get(size_t id) {
	return elapsed[id];
}


