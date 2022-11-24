#include <iostream>
#include <cmath>
#include "./ha_models/NNModel.hpp"
#include "./ha_models/layers/DenseLayer.hpp"
#include "./ha_models/TensorPool2D.hpp"

#include <chrono>
using namespace std::chrono;

// TODO: Network pools and inhertence of static variables

int main() {

	TENSOR_TYPE a[9] = { 1,0,0, 0,1,0, 0,0,1 };
	TENSOR_TYPE a1[9] = { 1,1,0, 0,0,0, 0,0,1 };
	TENSOR_TYPE b[9] = { 1,2,3,4,5,6,7,8,20 };
	TENSOR_TYPE c[9] = { 9,8,7,6,5,4,3,2,1 };
	TENSOR_TYPE res[9];

	//TensorPool2D t1(2, 3, 3);
	//t1.setValue(0,a);
	//t1.setValue(1, a1);

	TensorPool2D t2(2, 3, 3);
	t2.setValue(0, b);
	t2.setValue(1, c);

	Tensor2D t1(3, 3);
	t1.setValue(a);

	t2 = t2 * t1;

	Tensor2D sum(2, 1);

	t2.functionPass(Func::EXP);
	t2.sumAllElements(sum.getGpuPointer());
	t2.normalize(sum.getGpuPointer());

	gpuSync();

	t2.getValue(0, res);

	for (int i = 0; i < 9; i++) std::cout << res[i] << " ";

	return 0;

}