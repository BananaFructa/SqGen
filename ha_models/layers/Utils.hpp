#include "Layer.hpp"
#include "../../cuda_kernels/kernels.cuh"

Func activationToKernelFunc(Activation acv) {
	switch (acv) {
	case ReLU:
		return Func::KERNEL_ReLU;
	case SIGMOID:
		return Func::KERNEL_SIGMOID;
	case TANH:
		return Func::KERNEL_TANH;
	}
}