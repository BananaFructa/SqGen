#pragma once

#include "../cuda_kernels/kernels.cuh"
#include "Size.h"

#include "OperationDetails.hpp"

#include <cstdarg>

class Tensor {
public:

	static Tensor EmptyTensor;

	Size size;

	size_t mapSize = 0;
	size_t mapBlockSize = 0;
	size_t blockAllignOffset = 0;

	Tensor();

	Tensor(Size size);

	void free();

	virtual void init(Size size);

	// =============================================== 

	void setValue(Tensor_HOST t); // <------+
								  //		|--------- Dont support mapped tensors, change to kernel copy
	void getValue(Tensor_HOST t); // <------+

	// =============================================== 

	Scalar getElementAt(size_t pos,...);

	/*
	* @param begin = Start position on the last dimension
	* @param end = End position on the last dimension
	* @return A tensor object which represents a sub-memory region of the original tensor defined by an interval on its last dimension
	* (Ex: By slicing a 10x10 tensor from 2 to 5 it would result in a 10x3 tensor where the 1st, 2nd and 3rd columns 
	* are equal and references to the 2nd, 3rd and 4th columns from the initial tesnor.)
	*/
	Tensor slice(size_t begin, size_t end);

	Tensor_DEVICE getGpuDataPointer();

	TensorMap_DEVICE getGpuMapPointer();

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

protected:

	Tensor_DEVICE gpuTensorData = NULL_TENSOR;
	TensorMap_DEVICE gpuTensorMap = NULL_TENSOR;

	void setMap(TensorMap_HOST m);

};