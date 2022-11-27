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

	/*
	* @param begin = Start position on the last dimension
	* @param end = End position on the last dimension
	* @return A tensor object which represents a sub-memory region of the original tensor defined by an interval on its last dimension
	* (Ex: By slicing a 10x10 tensor from 2 to 5 it would result in a 10x3 tensor where the 1st, 2nd and 3rd columns 
	* are equal and references to the 2nd, 3rd and 4th columns from the initial tesnor.)
	*/
	Tensor slice(size_t begin, size_t end);

	Tensor_DEVICE getGpuPointer();

	void functionPass(Func f);

	void sumAllElements(Scalar sum);

	void sumAllElementsAcrossDim(Tensor& sums);

	void normalizeAcrossDim(Tensor& sums);

	void normalize(Scalar sum);


	const OperationDetails<Tensor, Tensor> operator*(Tensor& t);

	const OperationDetails<Tensor, Tensor> operator+(Tensor& t);

	const OperationDetails<Tensor, Tensor> operator%(Tensor& t);

	void operator=(const OperationDetails<Tensor, Tensor>& o);

	void operator+=(const OperationDetails<Tensor, Tensor>& o);

	void operator-=(const OperationDetails<Tensor, Tensor>& o);

private:

	Tensor_DEVICE gpuPointer = NULL_TENSOR;

};