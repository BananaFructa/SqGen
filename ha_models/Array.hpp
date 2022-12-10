#pragma once

#include "cuda_kernels/kernels.cuh"

template<typename T>
struct Array {
private:

	Array_DEVICE<T> gpuPointer;

public:

	size_t size = 0;

	Array() {

	}

	Array(size_t size) {
		init(size);
	}

	void init(size_t size) {
		gpuPointer = allocateArray<T>(size);
	}

	void free() {
		freeCudaMem(gpuPointer);
	}

	void setValue(Array_HOST<T> arr) {
		copyArrayFromHost<T>(arr, gpuPointer, size);
	}

	void getValue(Array_HOST<T> arr) {
		copyArrayFromDevice<T>(arr, gpuPointer, size);
	}

	Array<T> slice(size_t begin, size_t end) {
		Array<T> sliced;
		sliced.gpuPointer = this->gpuPointer + begin;
		sliced.size = end - begin;
		return sliced;
	}

};