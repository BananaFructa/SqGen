#include "TensorPool2D.hpp"

#include <cmath>

TensorPool2D TensorPool2D::EmptyPool = TensorPool2D();

TensorPool2D::TensorPool2D() {

}

TensorPool2D::TensorPool2D(size_t poolSize, size_t lines, size_t columns) {
	init(poolSize, lines, columns);
}

void TensorPool2D::free() {
	if (pool) freeTensor(pool);
}

void TensorPool2D::init(size_t poolSize, size_t lines, size_t columns) {
	if (!pool) {
		this->lines = lines;
		this->columns = columns;
		this->size = lines * columns * poolSize;
		this->poolSize = poolSize;
		this->elementSize = lines * columns;
		pool = allocateTensor(size);
	}
}

void TensorPool2D::setValue(size_t tensorId, TENSOR_TYPE t[]) {
	copyTensorFromHost((Tensor_HOST)t, pool + tensorId * elementSize, elementSize);
}

void TensorPool2D::getValue(size_t tensorId, TENSOR_TYPE t[]) {
	copyTensorFromDevice((Tensor_HOST)t, pool + tensorId * elementSize, elementSize);
}

Scalar TensorPool2D::getElementAt(size_t tensorId, size_t l, size_t c) {
	return pool + tensorId * elementSize + c + l * columns;
}

Tensor TensorPool2D::getGpuPointer() {
	return pool;
}

void TensorPool2D::functionPass(Func f) {
	CudaKernels::funcPass(pool, f, size);
}

void TensorPool2D::sumAllElements(Tensor sum) {
	CudaKernels::sumTensor(pool, sum, poolSize, elementSize);
}

void TensorPool2D::normalize(Tensor sum) {
	CudaKernels::normalizeTensor(pool, sum, poolSize, elementSize);
}

const OperationDetails<TensorPool2D, TensorPool2D> TensorPool2D::operator*(TensorPool2D& t) {
	return OperationDetails<TensorPool2D, TensorPool2D>(*this, t, Op::MUL2D);
}

const OperationDetails<TensorPool2D, TensorPool2D> TensorPool2D::operator+(TensorPool2D& t) {
	return OperationDetails<TensorPool2D, TensorPool2D>(*this, t, Op::SUM);
}

const OperationDetails<TensorPool2D, Tensor2D> TensorPool2D::operator*(Tensor2D& t)
{
	return OperationDetails<TensorPool2D, Tensor2D>(*this,t,Op::MUL2D);
}

const OperationDetails<TensorPool2D, Tensor2D> TensorPool2D::operator+(Tensor2D& t)
{
	return OperationDetails<TensorPool2D, Tensor2D>(*this,t,Op::SUM);
}

void TensorPool2D::operator=(const OperationDetails<TensorPool2D, Tensor2D>& o) {
	switch (o.operation) {
		case MUL2D:
			CudaKernels::mulTensor2D(pool, o.t1.pool, o.t2.getGpuPointer(), poolSize, lines, o.t1.columns, columns, true);
			break;
		case SUM:
			CudaKernels::addTensor(pool, o.t1.pool, o.t2.getGpuPointer(), elementSize, size, true);
			break;
	}
}

void TensorPool2D::operator=(const OperationDetails<TensorPool2D, TensorPool2D>& o) {
	switch (o.operation) {
		case MUL2D:
			CudaKernels::mulTensor2D(pool, o.t1.pool, o.t2.pool, std::min(o.t1.poolSize,o.t2.poolSize), lines, o.t1.columns, columns, false);
			break;
		case SUM:
			CudaKernels::addTensor(pool, o.t1.pool, o.t2.pool, elementSize, size, false);
			break;
	}
}
