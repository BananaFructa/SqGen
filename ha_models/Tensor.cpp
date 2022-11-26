#include "Tensor.hpp"

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

		if (size.dim > 1) {
			/* 
			* Just for the sake of better syntax when declaring tensors it is better to have for 2 dim or
			* larger tensors the 2D respective columns one after another in memory and not the lines
			*/
			if (i == 0) linearPos += size.getDimSize(1);
			if (i == 1) linearPos += size.getDimSize(0) * va_arg(argList, size_t);
		}
		else {
			size_t mul = 1;

			for (size_t j = 0; j < i; j++) mul *= size.getDimSize(j);

			linearPos += va_arg(argList, size_t) * mul;
		}

	}

	va_end(argList);

	return gpuPointer + linearPos;
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

void Tensor::operator=(const OperationDetails<Tensor, Tensor>& o) {
	switch (o.operation) {
		case MUL2D:
			CudaKernels::mulTensor2D(
				gpuPointer,
				o.t1.gpuPointer,
				o.t2.gpuPointer,
				o.t1.size.bidimensionalSize,
				o.t2.size.bidimensionalSize,
				o.t1.size.getDimSize(0),
				o.t1.size.getDimSize(1),
				o.t2.size.getDimSize(1)
			);
			break;
		case SUM:
			CudaKernels::addTensor(
				gpuPointer,
				o.t1.gpuPointer,
				o.t2.gpuPointer,
				o.t1.size.size,
				o.t2.size.size
			);
			break;
	}
}