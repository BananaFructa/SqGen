#pragma once

#include "cuda_kernels/kernels.cuh"
#include "Size.h"

#include "OperationDetails.hpp"

#include <cstdarg>

/*
*	  +-------+
	  |		  |
	  |		  V
	+---+ | +---+---+---+---+---+ ... +---+---+---+
	|M_0| | | 0	| 1	| 2	| 3	| 4	|	  |n-3|n-2|n-1|
	+---+ | +---+---+---+---+---+ ... +---+---+---+
			
			|<---------- one block -------------->|
*/

class Tensor {
public:

	static Tensor EmptyTensor;

	Size size;

	bool mapped = false;
	size_t mapSize = 0;
	size_t mapBlockSize = 0;
	size_t blockAllignOffset = 0;

	Tensor();

	/*
	* @brief Constructor which calls the Tensor::init() method with size as a parameter
	* @param size = The wanted size of the tensor
	*/
	Tensor(Size size);

	/*
	* @brief Frees all the GPU/CPU side memory associated with the tensor object.
	*/
	void free();

	/*
	* @brief Allocated the required GPU/CPU side memory and values
	* @param size = The wanted size of the tensor
	*/
	virtual void init(Size size);

	/*
	* @brief Sets the value of of the tensor to be equal to the input CPU side array
	* @param t = Input CPU side array
	*/
	void setValue(Tensor_HOST t);

	/*
	* @brief Gets the value of the tensor and copies it to a CPU side array
	* @param t = Input CPU side array
	*/
	void getValue(Tensor_HOST t);

	/*
	* @param pos = The position of the scalar
	* @return The GPU pointer of the scalar
	*/
	Scalar getElementAt(size_t pos,...);

	/*
	* @param begin = Start position on the last dimension
	* @param end = End position on the last dimension
	* @return A tensor object which represents a sub-memory region of the original tensor defined by an interval on its last dimension
	* (Ex: By slicing a 10x10 tensor from 2 to 5 it would result in a 10x3 tensor where the 1st, 2nd and 3rd columns 
	* are equal and references to the 2nd, 3rd and 4th columns from the initial tesnor.)
	*/
	Tensor slice(size_t begin, size_t end);

	/*
	* @return A tensor with the same data but with axis of lenght one removed
	*/
	Tensor squeeze();

	/*
	* @brief Clamps all the values between a defined bound
	* @param lower = Lower bound
	* @param upper = Upper bound
	*/
	void clamp(TENSOR_TYPE lower, TENSOR_TYPE upper);

	/*
	* @brief Performs a deep copy on the input tensor
	*/
	void copyTo(Tensor& t);

	/*
	* @return The GPU tensor data pointer
	*/
	Tensor_DEVICE getGpuDataPointer();

	/*
	* @return The GPU tensor map pointer
	*/
	TensorMap_DEVICE getGpuMapPointer();

	/*
	* @brief Initializes with 0 all values from the tensor
	*/
	void initZero();

	/*
	* @brief Applies a function to each element of the tensor
	* @param f = The function
	*/
	void functionPass(Func f);

	/*
	* @brief Summs all elements in the tensor and returns the result in sum
	*/
	void sumAllElements(Scalar sum);

	/*
	* @brief Summs all the elements in each sub-tensor defined by the indexing of the last dimension and return the results in the sums tensor
	*/
	void sumAllElementsAcrossDim(Tensor& sums);

	/*
	* @brief Normalizez all the elements in each sub-tensor defined by the indexing og the last dimesion
	* @param sums = The tensor containing the sums across dimentions
	*/
	void normalizeAcrossDim(Tensor& sums);

	/*
	* @brief Normalizez the whole tensor
	*/
	void normalize(Scalar sum);

	/*
	* @brief Syncs the tensor map between cpu and gpu side
	*/
	void syncMap();

	const OperationDetails<Tensor, Tensor> operator*(Tensor& t);

	const OperationDetails<Tensor, Tensor> operator+(Tensor& t);

	const OperationDetails<Tensor, Tensor> operator%(Tensor& t);

	void operator=(const OperationDetails<Tensor, Tensor>& o);

	void operator+=(const OperationDetails<Tensor, Tensor>& o);

	void operator-=(const OperationDetails<Tensor, Tensor>& o);

	/*
	* @brief Saves the tensor in a .npy file
	* @param path = The file path where to save the tensor
	*/
	void save(const char* path);

	/*
	* @brief Load the tensor from a .npy file
	* @param path = The file path from where to load the tensor
	*/
	void load(const char* path);

protected:

	bool referenceOnly = false;

	Tensor_DEVICE gpuTensorData = NULL_TENSOR;
	TensorMap_DEVICE gpuTensorMap = NULL_TENSOR_MAP;
	TensorMap_HOST hostMap = NULL_TENSOR_MAP;

	void setMap(TensorMap_HOST m);

};