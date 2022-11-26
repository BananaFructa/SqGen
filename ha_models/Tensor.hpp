#pragma once

#include "../cuda_kernels/kernels.cuh"
#include "Size.h"

#include "OperationDetails.hpp"

#include <cstdarg>

struct Tensor {
public:

	static Tensor EmptyTensor;

	Size size;

	Tensor();

	Tensor(Size size);

	void free();

	void init(Size size);

	void setValue(TENSOR_TYPE t[]);

	void getValue(TENSOR_TYPE t[]);

	Scalar getElementAt(size_t pos,...);

	Tensor_DEVICE getGpuPointer();

	void functionPass(Func f);

	void sumAllElements(Scalar sum);

	void sumAllElementsAcrossDim(Tensor& sums);

	void normalizeAcrossDim(Tensor& sums);

	void normalize(Scalar sum);


	const OperationDetails<Tensor, Tensor> operator*(Tensor& t);

	const OperationDetails<Tensor, Tensor> operator+(Tensor& t);

	void operator=(const OperationDetails<Tensor, Tensor>& o);

private:

	Tensor_DEVICE gpuPointer = NULL_TENSOR;

};