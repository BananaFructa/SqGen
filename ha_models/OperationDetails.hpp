#pragma once

#include "../cuda_kernels/kernels.cuh"

#include <iostream>

enum Op {
	MUL2D,
	SUM
};

template<typename T, typename U>
struct OperationDetails {
public:

	T& t1;
	U& t2;

	Op operation;

	OperationDetails(T& t1, U& t2, Op operation) : t1(t1) , t2(t2) {
		this->operation = operation;
	}

};