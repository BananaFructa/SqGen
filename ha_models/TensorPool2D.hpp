#pragma once

#include "../cuda_kernels/kernels.cuh"

#include "OperationDetails.hpp"

#include "Tensor2D.hpp"

struct TensorPool2D {
public:

	static TensorPool2D EmptyPool;

	size_t poolSize = 0;
	size_t lines = 0;
	size_t columns = 0;
	size_t size = 0;
	size_t elementSize = 0;

	TensorPool2D();

	TensorPool2D(size_t poolSize, size_t lines, size_t columns);

	void free();

	void init(size_t poolSize, size_t lines, size_t columns);

	void setValue(size_t tensorId, TENSOR_TYPE t[]);

	void getValue(size_t tensorId, TENSOR_TYPE t[]);

	Scalar getElementAt(size_t tensorId, size_t l, size_t c);

	Tensor getGpuPointer();

	void functionPass(Func f);

	void sumAllElements(Tensor sum);

	void normalize(Tensor sum);

	const OperationDetails<TensorPool2D,TensorPool2D> operator*(TensorPool2D& t);

	const OperationDetails<TensorPool2D, TensorPool2D> operator+(TensorPool2D& t);

	const OperationDetails<TensorPool2D, Tensor2D> operator*(Tensor2D& t);

	const OperationDetails<TensorPool2D, Tensor2D> operator+(Tensor2D& t);

	void operator=(const OperationDetails<TensorPool2D, TensorPool2D>& o);

	void operator=(const OperationDetails<TensorPool2D, Tensor2D>& o);

private:

	Tensor pool = NULL_TENSOR;

};