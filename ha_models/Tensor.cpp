#include "Tensor.hpp"

// Too many paramters so these are some functions to wrap everything up
void callMulKernerl(Tensor& tt, Tensor& t1, Tensor& t2, int operand) {
	if (tt.mapSize == 1 && t1.mapSize == 1 && t2.mapSize == 1) {
		CudaKernels::mulTensor2D(
			tt.getGpuDataPointer(),
			t1.getGpuDataPointer(),
			t2.getGpuDataPointer(),
			t1.size.bidimensionalSize,
			t2.size.bidimensionalSize,
			t1.size.getDimSize(0),
			t1.size.getDimSize(1),
			t2.size.getDimSize(1),
			operand
		);
	}
	else {
		CudaKernels::mulTensorMapped2D(
			tt.getGpuMapPointer(),
			t1.getGpuMapPointer(),
			t2.getGpuMapPointer(),
			t1.size.bidimensionalSize,
			t2.size.bidimensionalSize,
			t1.size.getDimSize(0),
			t1.size.getDimSize(1),
			t2.size.getDimSize(1),
			tt.mapBlockSize,
			t1.mapBlockSize,
			t2.mapBlockSize,
			tt.blockAllignOffset,
			t1.blockAllignOffset,
			t2.blockAllignOffset,
			operand
		);
	}
}

void callAddKernel(Tensor& tt, Tensor& t1, Tensor& t2, int operand) {
	if (tt.mapSize == 1 && t1.mapSize == 1 && t2.mapSize == 1) {
		CudaKernels::addTensor(
			tt.getGpuDataPointer(),
			t1.getGpuDataPointer(),
			t2.getGpuDataPointer(),
			t1.size.size,
			t2.size.size,
			operand
		);
	}
	else {
		CudaKernels::addTensorMapped(
			tt.getGpuMapPointer(),
			t1.getGpuMapPointer(),
			t2.getGpuMapPointer(),
			t1.size.size,
			t2.size.size,
			tt.mapBlockSize,
			t1.mapBlockSize,
			t2.mapBlockSize,
			tt.blockAllignOffset,
			t1.blockAllignOffset,
			t2.blockAllignOffset,
			operand
		);
	}
}

void callHadmaradKernel(Tensor& tt, Tensor& t1, Tensor& t2, int operand) {
	if (tt.mapSize == 1 && t1.mapSize == 1 && t2.mapSize == 1) {
		CudaKernels::hadamardTensor(
			tt.getGpuDataPointer(),
			t1.getGpuDataPointer(),
			t2.getGpuDataPointer(),
			t1.size.size,
			t2.size.size,
			operand
		);
	}
	else {
		CudaKernels::hadamardTensorMapped(
			tt.getGpuMapPointer(),
			t1.getGpuMapPointer(),
			t2.getGpuMapPointer(),
			t1.size.size,
			t2.size.size,
			tt.mapBlockSize,
			t1.mapBlockSize,
			t2.mapBlockSize,
			tt.blockAllignOffset,
			t1.blockAllignOffset,
			t2.blockAllignOffset,
			operand
		);
	}
}

Tensor Tensor::EmptyTensor = Tensor();

Tensor::Tensor() {

}

Tensor::Tensor(Size size) {
	init(size);
}

void Tensor::free() {
	if (gpuTensorMap) freeCudaMem(gpuTensorMap);
	else if (gpuTensorData) freeCudaMem(gpuTensorData);
}

void Tensor::init(Size size) {
	if (!gpuTensorData) {
		this->size = size;
		this->mapSize = 1;
		this->mapBlockSize = size.size / mapSize;
		AllocRes res = allocateTensor(size.size,mapSize);
		gpuTensorData = res.data;
		gpuTensorMap = res.map;

		TENSOR_TYPE* map[] = { gpuTensorData };
		setMap((TensorMap_HOST)map);
	}
}

void Tensor::setValue(Tensor_HOST t) {
	copyTensorFromHost(t, gpuTensorData, size.size);
}

void Tensor::getValue(Tensor_HOST t) {
	copyTensorFromDevice(t, gpuTensorData, size.size);
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

	return gpuTensorData + linearPos;
}

Tensor Tensor::slice(size_t begin, size_t end) {
	size_t subSize = size.size / size.getDimSize(size.dim - 1);
	size_t* sizes = new size_t[size.dim];
	for (int i = 0; i < size.dim - 1; i++) sizes[i] = size.getDimSize(i);
	sizes[size.dim - 1] = end - begin;
	Tensor sliced;
	sliced.gpuTensorData = gpuTensorData + subSize * begin;
	sliced.gpuTensorMap = gpuTensorMap + begin / mapBlockSize;
	sliced.size = Size(size.dim, sizes);
	sliced.mapSize = 1 + (end - 1) / mapBlockSize - begin / mapBlockSize;
	sliced.mapBlockSize = mapBlockSize;
	sliced.blockAllignOffset = begin % mapBlockSize;
	return sliced;
}

Tensor_DEVICE Tensor::getGpuDataPointer() {
	return gpuTensorData;
}

TensorMap_DEVICE Tensor::getGpuMapPointer() {
	return gpuTensorMap;
}

void Tensor::functionPass(Func f) {
	CudaKernels::funcPass(gpuTensorData, f, size.size);
}

void Tensor::sumAllElements(Scalar sum) {
	CudaKernels::sumTensor(gpuTensorData, (Tensor_DEVICE)sum, 1, size.size);
}

void Tensor::normalize(Scalar sum) {
	CudaKernels::normalizeTensor(gpuTensorData, (Tensor_DEVICE)sum, 1, size.size);
}

void Tensor::sumAllElementsAcrossDim(Tensor& sums) {
	size_t last = size.getDimSize(size.dim - 1);
	CudaKernels::sumTensor(gpuTensorData, sums.getGpuDataPointer(), last, size.size / last);
}

void Tensor::normalizeAcrossDim(Tensor& sum) {
	size_t last = size.getDimSize(size.dim - 1);
	CudaKernels::normalizeTensor(gpuTensorData, sum.getGpuDataPointer(), last, size.size / last);
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

void Tensor::setMap(TensorMap_HOST m) {
	copyMapFromHost(m, gpuTensorMap, mapSize);
}
