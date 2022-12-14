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
	if (mapSize == 1) {
		copyTensorFromHost(t, hostMap[0] + blockAllignOffset, size.size);
		return;
	}
	size_t accumulator = 0;
	for (int i = 0; i < mapSize; i++) {
		if (i == 0) {
			copyTensorFromHost(t + accumulator, hostMap[i] + blockAllignOffset, mapBlockSize - blockAllignOffset);
		}
		else if (i == mapSize - 1) {
			size_t endOffset = (size.size + blockAllignOffset) % mapSize;
			copyTensorFromHost(t + accumulator, hostMap[i], endOffset);
		}
		else {
			copyTensorFromHost(t + accumulator, hostMap[i], mapBlockSize);
		}
		if (i == 0) accumulator += blockAllignOffset;
		else accumulator += mapBlockSize;
	}
}

// 

void Tensor::getValue(Tensor_HOST t) {
	// Quite sure this can be done better but yea...
	if (mapSize == 1) {
		copyTensorFromDevice(t, hostMap[0] + blockAllignOffset, size.size);
		return;
	}
	size_t accumulator = 0;
	for (int i = 0; i < mapSize; i++) {
		if (i == 0) {
			copyTensorFromDevice(t + accumulator, hostMap[i] + blockAllignOffset, mapBlockSize - blockAllignOffset);
		}
		else if (i == mapSize - 1) {
			size_t endOffset = (size.size + blockAllignOffset) % mapSize;
			copyTensorFromDevice(t + accumulator, hostMap[i], endOffset);
		}
		else {
			copyTensorFromDevice(t + accumulator, hostMap[i], mapBlockSize);
		}
		if (i == 0) accumulator += blockAllignOffset;
		else accumulator += mapBlockSize;
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

	size_t subSize = size.size / size.last();
	size_t* sizes = new size_t[size.dim];
	for (int i = 0; i < size.dim - 1; i++) sizes[i] = size.getDimSize(i);
	sizes[size.dim - 1] = end - begin;

	Tensor sliced;

	sliced.size = Size(size.dim, sizes);

	size_t endBlockId = ((end - 1) * subSize + blockAllignOffset) / mapBlockSize;
	size_t beginBlockId = (begin * subSize + blockAllignOffset) / mapBlockSize;
	size_t beginBlockIndex = (begin * subSize + blockAllignOffset) % mapBlockSize;

	sliced.mapSize = 1 + endBlockId - beginBlockId;

	//if (sliced.mapSize == 1) sliced.mapBlockSize = (end - begin) * subSize;
	/*else*/ sliced.mapBlockSize = mapBlockSize;

	sliced.blockAllignOffset = beginBlockIndex;

	sliced.gpuTensorMap = gpuTensorMap + beginBlockId;
	sliced.hostMap = hostMap + beginBlockId;

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


Tensor Tensor::squeeze() {
	Tensor squeezed = *this;
	squeezed.size = size.squeeze();
	return squeezed;
}


void Tensor::copyTo(Tensor& t) {
	copyTensorD2D(t.getGpuMapPointer(), gpuTensorMap, mapSize, size.size);
}

void Tensor::clamp(TENSOR_TYPE lower, TENSOR_TYPE upper) {

	if (!referenceOnly) CudaKernels::clampTensor(gpuTensorData, size.size, lower, upper);
	else CudaKernels::clampTensorMapped(gpuTensorMap, size.size, mapBlockSize, blockAllignOffset, lower, upper);

}

Tensor_DEVICE Tensor::getGpuDataPointer() {
	return gpuTensorData;
}


TensorMap_DEVICE Tensor::getGpuMapPointer() {
	return gpuTensorMap;
}


void Tensor::initZero() {
	if (!referenceOnly) CudaKernels::initZeroTensor(gpuTensorData, size.size);
	else CudaKernels::initZeroTensorMapped(gpuTensorMap, size.size, mapBlockSize, blockAllignOffset);
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