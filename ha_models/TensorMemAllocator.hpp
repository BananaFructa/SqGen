#pragma once

#include <vector>
#include <utility>
#include <concepts>

#include "../ha_models/Size.h"
#include "../ha_models/Tensor.hpp"

template<typename T>
struct TensorMemAllocator {
private:

	Size size;

	std::vector<T> unusedPool;

public:

	TensorMemAllocator(Size size) {
		this->size = size;
	}

	T getTensor() {
		bool reuse = !unusedPool.empty();

		T set;

		if (reuse) {
			set = unusedPool[unusedPool.size() - 1];
			unusedPool.pop_back();
		}
		else {
			set.init(size);
		}

		return set;
	}

	void freeTensor(T t) {
		unusedPool.push_back(t);
	}


};

