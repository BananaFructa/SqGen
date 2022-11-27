#include "Tensor.hpp"

// Too many paramters so these are some functions to wrap everything up
void callMulKernerl(Tensor& tt, Tensor& t1, Tensor& t2, int operand) {
	CudaKernels::mulTensor2D(
		tt.getGpuPointer(),
		t1.getGpuPointer(),
		t2.getGpuPointer(),
		t1.size.bidimensionalSize,
		t2.size.bidimensionalSize,
		t1.size.getDimSize(0),
		t1.size.getDimSize(1),
		t2.size.getDimSize(1),
		operand
	);
}

void callAddKernel(Tensor& tt, Tensor& t1, Tensor& t2, int operand) {
	CudaKernels::addTensor(
		tt.getGpuPointer(),
		t1.getGpuPointer(),
		t2.getGpuPointer(),
		t1.size.size,
		t2.size.size,
		operand
	);
}

void callHadmaradKernel(Tensor& tt, Tensor& t1, Tensor& t2, int operand) {
	CudaKernels::hadamardTensor(
		tt.getGpuPointer(),
		t1.getGpuPointer(),
		t2.getGpuPointer(),
		t1.size.size,
		t2.size.size,
		operand
	);
}

Tensor Tensor::EmptyTensor = Tensor();

Tensor::Tensor() {

}

Tensor::Tensor(Size size) {
	this->size = size;
	gpuPointer = allocateTensor(size.size);
}

void Tensor::free() {
	if (gpuPointer) freeTensor(gpuPointer);
}

void Tensor::init(Size size) {
	if (!gpuPointer) {
		this->size = size;
		gpuPointer = allocateTensor(size.size);
	}
}

void Tensor::setValue(TENSOR_TYPE t[]) {
	copyTensorFromHost((Tensor_HOST)t, gpuPointer, size.size);
}

void Tensor::getValue(TENSOR_TYPE t[]) {
	copyTensorFromDevice((Tensor_HOST)t, gpuPointer, size.size);
}

Scalar Tensor::getElementAt(size_t pos,...) {

	std::va_list argList;
	va_start(argList, pos);

	size_t linearPos = 0;

	for (size_t i = 0; i < size.dim; i++) {

		size_t mul = 1;

		for (size_t j = 0; j < i; j++) mul *= size.getDimSize(j);

		linearPos += va_arg(argList, size_t) * mul;
	}

	va_end(argList);

	return gpuPointer + linearPos;
}

Tensor Tensor::slice(size_t begin, size_t end) {
	size_t subSize = size.size / size.getDimSize(size.dim - 1);
	size_t* sizes = new size_t[size.dim];
	for (int i = 0; i < size.dim - 1; i++) sizes[i] = size.getDimSize(i);
	sizes[size.dim - 1] = end - begin;
	Tensor sliced;
	sliced.gpuPointer = gpuPointer + subSize * begin;
	sliced.size = Size(size.dim, sizes);
	return sliced;
}

Tensor_DEVICE Tensor::getGpuPointer() {
	return gpuPointer;
}

void Tensor::functionPass(Func f) {
	CudaKernels::funcPass(gpuPointer, f, size.size);
}

void Tensor::sumAllElements(Scalar sum) {
	CudaKernels::sumTensor(gpuPointer, (Tensor_DEVICE)sum, 1, size.size);
}

void Tensor::normalize(Scalar sum) {
	CudaKernels::normalizeTensor(gpuPointer, (Tensor_DEVICE)sum, 1, size.size);
}

void Tensor::sumAllElementsAcrossDim(Tensor& sums) {
	size_t last = size.getDimSize(size.dim - 1);
	CudaKernels::sumTensor(gpuPointer, sums.getGpuPointer(), last, size.size / last);
}

void Tensor::normalizeAcrossDim(Tensor& sum) {
	size_t last = size.getDimSize(size.dim - 1);
	CudaKernels::normalizeTensor(gpuPointer, sum.getGpuPointer(), last, size.size / last);
}

const OperationDetails<Tensor,Tensor> Tensor::operator*(Tensor& t) {
	return OperationDetails<Tensor, Tensor>(*this, t, Op::MUL2D);
}

const OperationDetails<Tensor, Tensor> Tensor::operator+(Tensor& t) {
	return OperationDetails<Tensor, Tensor>(*this, t, Op::SUM);
}

const OperationDetails<Tensor, Tensor> Tensor::operator%(Tensor& t)
{
	return OperationDetails<Tensor, Tensor>(*this, t, Op::HADAMARD);
}

void Tensor::operator=(const OperationDetails<Tensor, Tensor>& o) {
	switch (o.operation) {
		case MUL2D:
			callMulKernerl(*this, o.t1, o.t2, 0);
			break;
		case SUM:
			callAddKernel(*this, o.t1, o.t2, 0);
			break;
		case HADAMARD:
			callHadmaradKernel(*this, o.t1, o.t2, 0);
			break;
	}
}

void Tensor::operator+=(const OperationDetails<Tensor, Tensor>& o) {
	switch (o.operation) {
		case MUL2D:
			callMulKernerl(*this, o.t1, o.t2, 1);
			break;
		case SUM:
			callAddKernel(*this, o.t1, o.t2, 1);
			break;
		case HADAMARD:
			callHadmaradKernel(*this, o.t1, o.t2, 1);
			break;
	}
}

void Tensor::operator-=(const OperationDetails<Tensor, Tensor>& o) {
	switch (o.operation) {
		case MUL2D:
			callMulKernerl(*this, o.t1, o.t2, -1);
			break;
		case SUM:
			callAddKernel(*this, o.t1, o.t2, -1);
			break;
		case HADAMARD:
			callHadmaradKernel(*this, o.t1, o.t2, -1);
			break;
	}
}
