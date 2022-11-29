#pragma once

#include <memory>
#include <cstdarg>

struct Size {
public:

	std::shared_ptr<size_t[]> sizes;
	size_t dim = 0;
	size_t size = 0;
	size_t bidimensionalSize = 0;

	Size() {
		
	}

	Size(size_t s, ...) {
		dim = s;
		sizes = std::shared_ptr<size_t[]>(new size_t[dim]);
		std::va_list argList;
		va_start(argList, s);

		for (size_t i = 0; i < dim; i++) {
			size_t a = va_arg(argList, size_t);
			sizes.get()[i] = a;
		}

		va_end(argList);

		calConstants();
	}

	// For user friendly interfacing
	Size(int s, ...) {
		dim = s;
		sizes = std::shared_ptr<size_t[]>(new size_t[dim]);
		std::va_list argList;
		va_start(argList, s);

		for (size_t i = 0; i < dim; i++) {
			size_t a = va_arg(argList, int);
			sizes.get()[i] = a;
		}

		va_end(argList);

		calConstants();
	}

	Size(size_t dim, size_t* s) {
		this->dim = dim;
		sizes = std::shared_ptr<size_t[]>(s);
		calConstants();
	}

	size_t getDim() {
		return dim;
	}

	size_t getDimSize(size_t d) {
		if (d >= dim) return 1;
		return sizes.get()[d];
	}

	size_t last() {
		return getDimSize(dim - 1);
	}

private:

	void calConstants() {
		if (dim > 0) size = 1;
		if (dim > 1) bidimensionalSize = 1;

		for (size_t i = 0; i < dim; i++) {
			if (i > 1) {
				bidimensionalSize *= sizes.get()[i];
			}
			size_t a = sizes.get()[i];
			size *= sizes.get()[i];
		}
	}

};