#include "Tensor2D.hpp"

Tensor2D Tensor2D::EmptyTensor = Tensor2D();

Tensor2D::Tensor2D() {

}

Tensor2D::Tensor2D(size_t lines, size_t columns) {
	init(lines, columns);
}

void Tensor2D::free() {
	if (gpuPointer) freeTensor(gpuPointer);
}

void Tensor2D::init(size_t lines, size_t columns) {
	if (!gpuPointer) {
		this->lines = lines;
		this->columns = columns;
		this->size = lines * columns;
		gpuPointer = allocateTensor(size);
	}
}

void Tensor2D::setValue(TENSOR_TYPE t[]) {
	copyTensorFromHost((Tensor_HOST)t, gpuPointer, size);
}

void Tensor2D::getValue(TENSOR_TYPE t[]) {
	copyTensorFromDevice((Tensor_HOST)t, gpuPointer, size);
}

Scalar Tensor2D::getElementAt(size_t l, size_t c) {
	return gpuPointer + c + l * columns;
}

Tensor Tensor2D::getGpuPointer() {
	return gpuPointer;
}

void Tensor2D::functionPass(Func f) {
	funcPass(gpuPointer, f, size);
}

void Tensor2D::sumAllElements(Scalar sum) {
	sumTensor(gpuPointer, (Tensor)sum, 1, size);
}

void Tensor2D::normalize(Scalar sum) {
	normalizeTensor(gpuPointer, (Tensor)sum, 1, size);
}

const OperationDetails<Tensor2D,Tensor2D> Tensor2D::operator*(Tensor2D& t) {
	return OperationDetails<Tensor2D, Tensor2D>(*this, t, Op::MUL2D);
}

const OperationDetails<Tensor2D, Tensor2D> Tensor2D::operator+(Tensor2D& t) {
	return OperationDetails<Tensor2D, Tensor2D>(*this, t, Op::SUM);
}

void Tensor2D::operator=(const OperationDetails<Tensor2D, Tensor2D>& o) {
	switch (o.operation) {
		case MUL2D:
			mulTensor2D(gpuPointer, o.t1.gpuPointer, o.t2.gpuPointer, 1, lines, o.t1.columns, columns, true);
			break;
		case SUM:
			addTensor(gpuPointer, o.t1.gpuPointer, o.t2.gpuPointer, size, size, true);
			break;
	}
}