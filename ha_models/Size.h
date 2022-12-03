#pragma once

#include <vector>
#include <cstdarg>

struct Size {
public:

	std::vector<size_t> sizes;
	size_t dim = 0;
	size_t size = 0;
	size_t bidimensionalSize = 0;

	Size() {
		
	}

	Size(const Size& s) {
		this->dim = s.dim;
		this->size = s.size;
		this->bidimensionalSize = s.bidimensionalSize;
		this->sizes = std::vector<size_t>(s.sizes);
	}

	Size(size_t s, ...) {
		dim = s;
		std::va_list argList;
		va_start(argList, s);

		for (size_t i = 0; i < dim; i++) {
			size_t a = va_arg(argList, size_t);
			sizes.push_back(a);
		}

		va_end(argList);

		calConstants();
	}

	// For user friendly interfacing
	Size(int s, ...) {
		dim = s;
		std::va_list argList;
		va_start(argList, s);

		for (size_t i = 0; i < dim; i++) {
			size_t a = va_arg(argList, int);
			sizes.push_back(a);
		}

		va_end(argList);

		calConstants();
	}

	Size(size_t dim, size_t* s) {
		this->dim = dim;
		for (size_t i = 0; i < dim; i++) sizes.push_back(s[i]);
		calConstants();
	}

	size_t getDim() {
		return dim;
	}

	size_t getDimSize(size_t d) {
		if (d >= dim) return 1;
		return sizes[d];
	}

	size_t last() {
		return getDimSize(dim - 1);
	}

	void extend(size_t s) {
		dim++;
		size *= s;
		if (dim == 2) bidimensionalSize = 1;
		else if (dim > 2) bidimensionalSize *= s;
		sizes.push_back(s);
	}

private:

	void calConstants() {
		if (dim > 0) size = 1;
		if (dim > 1) bidimensionalSize = 1;

		for (size_t i = 0; i < dim; i++) {
			if (i > 1) {
				bidimensionalSize *= sizes[i];
			}
			size_t a = sizes[i];
			size *= sizes[i];
		}
	}

};