#pragma once

#include "../../cuda_kernels/kernels.cuh"

#include <vector>
#include <thread>

struct AllocatedInterval {
public:
	size_t start;
	size_t end;
};

struct CurandManager {

	cudaStream_t randStream;
	curandState_t* curandStatePool;

	std::vector<AllocatedInterval> allocatedIntervals;



};