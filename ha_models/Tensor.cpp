#include "Tensor.hpp"

// Too many paramters so these are some functions to wrap everything up
void callMulKernerl(Tensor& tt, Tensor& t1, Tensor& t2, int operand) {
	if (!tt.mapped && !t1.mapped == 1 && !t2.mapped) {
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
	if (!tt.mapped && !t1.mapped && !t2.mapped) {
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
	if (!tt.mapped && !t1.mapped && !t2.mapped) {
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
	if (hostMap) delete[] hostMap;
}

void Tensor::init(Size size) {
	if (!gpuTensorData) {
		this->size = size;
		this->mapSize = 1;
		this->mapBlockSize = size.size / mapSize;
		AllocRes res = allocateTensor(size.size,mapSize);
		gpuTensorData = res.data;
		gpuTensorMap = res.map;

		hostMap = new Tensor_DEVICE[mapSize];
		hostMap[0] = gpuTensorData;
		syncMap();
	}
}

void Tensor::setValue(Tensor_HOST t) {
	for (int i = 0; i < mapSize; i++) {
		copyTensorFromHost(t + i * mapBlockSize, hostMap[i], mapBlockSize);
	}
}

void Tensor::getValue(Tensor_HOST t) {
	for (int i = 0; i < mapSize; i++) {
		copyTensorFromDevice(t + i * mapBlockSize, hostMap[i], mapBlockSize);
	}
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

	return gpuTensorMap[linearPos / mapBlockSize] + linearPos % mapBlockSize;
}

Tensor Tensor::slice(size_t begin, size_t end) {

	// TAKE BLOCK OFFSET INTO ACCOUNT AAAAAAAAAAAAAAAAAAAA

	size_t subSize = size.size / size.last();
	size_t* sizes = new size_t[size.dim];
	for (int i = 0; i < size.dim - 1; i++) sizes[i] = size.getDimSize(i);
	sizes[size.dim - 1] = end - begin;

	Tensor sliced;

	sliced.size = Size(size.dim, sizes);

	sliced.mapSize = 1 + (end - 1) * subSize / mapBlockSize - begin * subSize / mapBlockSize;
	sliced.mapBlockSize = mapBlockSize;
	sliced.blockAllignOffset = begin * subSize % mapBlockSize;

	sliced.gpuTensorMap = gpuTensorMap + begin * subSize / mapBlockSize;
	sliced.hostMap = hostMap + begin * subSize / mapBlockSize;

	if (!referenceOnly && mapSize == 1) {
		sliced.gpuTensorData = gpuTensorData + subSize * begin;
		sliced.referenceOnly = false;
		sliced.mapped = false;
	}
	else {
		sliced.referenceOnly = true;
		sliced.mapped = true;
	}

	return sliced;
}

Tensor_DEVICE Tensor::getGpuDataPointer() {
	return gpuTensorData;
}

TensorMap_DEVICE Tensor::getGpuMapPointer() {
	return gpuTensorMap;
}

void Tensor::functionPass(Func f) {
	if (!mapped) CudaKernels::funcPass(gpuTensorData, f, size.size);
	else CudaKernels::funcPassMapped(gpuTensorMap, mapBlockSize, blockAllignOffset, size.size, f);
}

void Tensor::sumAllElements(Scalar sum) {
	if (!mapped) CudaKernels::sumTensor(gpuTensorData, (Tensor_DEVICE)sum, 1, size.size);
	else CudaKernels::sumTensorMapped(gpuTensorMap, (Tensor_DEVICE)sum, 1, size.size, mapBlockSize, blockAllignOffset);
}

void Tensor::normalize(Scalar sum) {
	if (!mapped) CudaKernels::normalizeTensor(gpuTensorData, (Tensor_DEVICE)sum, 1, size.size);
	else CudaKernels::normalizeTensorMapped(gpuTensorMap, (Tensor_DEVICE)sum, 1, size.size, mapBlockSize, blockAllignOffset);
}

void Tensor::syncMap() {
	setMap(hostMap);
}

void Tensor::sumAllElementsAcrossDim(Tensor& sums) {
	size_t last = size.last();
	if (!mapped) CudaKernels::sumTensor(gpuTensorData, sums.getGpuDataPointer(), last, size.size / last);
	else CudaKernels::sumTensorMapped(gpuTensorMap, sums.getGpuDataPointer(), last, size.size / last, mapBlockSize, blockAllignOffset);
}

void Tensor::normalizeAcrossDim(Tensor& sum) {
	size_t last = size.last();
	if (!mapped) CudaKernels::normalizeTensor(gpuTensorData, sum.getGpuDataPointer(), last, size.size / last);
	else CudaKernels::normalizeTensorMapped(gpuTensorMap, sum.getGpuDataPointer(), last, size.size / last, mapBlockSize, blockAllignOffset);
}

const OperationDetails<Tensor,Tensor> Tensor::operator*(Tensor& t) {
	return OperationDetails<Tensor, Tensor>(*this, t, Op::MUL2D);
}

const OperationDetails<Tensor, Tensor> Tensor::operator+(Tensor& t) {
	return OperationDetails<Tensor, Tensor>(*this, t, Op::SUM);
}

const OperationDetails<Tensor, Tensor> Tensor::operator%(Tensor& t) {
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
