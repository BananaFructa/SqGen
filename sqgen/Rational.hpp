#pragma once

#include <algorithm>

struct Rational {
	long long a = 0;
	long long b = 1;

	const float toFloat() const {
		return (float)a / b;
	}

	const Rational operator+(Rational r) const {

		if (b == r.b) {
			return { a + r.a, b };
		}
		if (b > r.b && (b % r.b) == 0) {
			return { a + r.a * (b / r.b), b };
		}
		if (b < r.b && (r.b % b) == 0) {
			return { a * (r.b / b) + r.a, r.b };
		}
		return { a * r.b + r.a * b, b * r.b };
	}

	const Rational operator-() const {
		return { -a, b };
	}

	const Rational operator-(const Rational r) const {
		return -r + *this;
	}

	const Rational operator*(const Rational r) const {
		long long a_ = a * r.a;
		long long b_ = b;
		if (a_ % r.b == 0) {
			a_ /= r.b;
		}
		else if (a_ % b == 0) {
			a_ /= b;
			b_ = r.b;
		}
		else b_ *= r.b;
		return { a_, b_ };
	}

	const Rational operator+=(const Rational r) {
		*this = *this + r;
		return *this;
	}

	const Rational operator/(const Rational r) {
		return *this * r.inv();
	}

	const Rational inv() const {
		return { b, a };
	}

	const bool operator==(const Rational r) const {
		return a * r.b == b * r.a;
	}

	const bool operator>(const Rational r) const {
		return toFloat() > r.toFloat();
	}

	const bool operator<(const Rational r) const {
		return toFloat() < r.toFloat();
	}

	const bool operator>=(const Rational r) const {
		return *this > r || *this == r;
	}

	const bool operator<=(const Rational r) const {
		return *this < r || *this == r;
	}

	const bool negative() const {
		return a < 0;
	}

	void multiply(long long m) {
		a *= m;
		b *= m;
	}

};