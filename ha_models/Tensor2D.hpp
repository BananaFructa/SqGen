#pragma once

#include "../cuda_kernels/kernels.cuh"

#include "OperationDetails.hpp"

struct Tensor2D {
public:

	static Tensor2D EmptyTensor;

	size_t lines = 0;
	size_t columns = 0;
	size_t size = 0;

	Tensor2D();

	Tensor2D(size_t lines, size_t columns);

	void free();

	void init(size_t lines, size_t columns);

	void setValue(TENSOR_TYPE t[]);

	void getValue(TENSOR_TYPE t[]);

	Scalar getElementAt(size_t l, size_t c);

	Tensor getGpuPointer();

	void functionPass(Func f);

	void sumAllElements(Scalar sum);

	void normalize(Scalar sum);


	const OperationDetails<Tensor2D, Tensor2D> operator*(Tensor2D& t);

	const OperationDetails<Tensor2D, Tensor2D> operator+(Tensor2D& t);

	void operator=(const OperationDetails<Tensor2D, Tensor2D>& o);

private:

	Tensor gpuPointer = NULL_TENSOR;

};