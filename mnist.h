#ifndef MNIST_H
#define MNIST_H

#include <cmath>
#include <map>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

class Matrix {
public:
	typedef float real_t;
private:
	struct Data {
		size_t refs = 0;
		size_t rows = 0;
		size_t cols = 0;
		std::vector<real_t> vals;
	};
	Data *d = nullptr;
	void assign(Data *p)
	{
		if (p != d) {
			if (p) {
				p->refs++;
			}
			if (d) {
				if (d->refs > 1) {
					d->refs--;
				} else {
					delete d;
				}
			}
			d = p;
		}
	}
public:

	Matrix() = default;
	~Matrix()
	{
		assign(nullptr);
	}
	Matrix(Matrix const &r)
	{
		assign(r.d);
	}
	void operator = (Matrix const &r)
	{
		assign(r.d);
	}

	Matrix copy() const
	{
		Matrix t;
		if (d) {
			Data *p = new Data(*d);
			p->refs = 0;
			t.assign(p);
		}
		return t;
	}

	void copy_on_write()
	{
		if (d) {
			if (d->refs > 1) {
				Data *p = new Data(*d);
				p->refs = 0;
				assign(p);
			}
		}
	}

	bool empty() const
	{
		return (d ? d->vals.size() : 0) == 0;
	}

	size_t size() const
	{
		return d->vals.size();
	}

	real_t *data()
	{
		return d->vals.data();
	}

	real_t const *data() const
	{
		return d->vals.data();
	}

	size_t rows() const
	{
		return d->rows;
	}

	size_t cols() const
	{
		return d->cols;
	}

	void make(size_t r, size_t c);
	void make(size_t r, size_t c, std::initializer_list<real_t> const &list);
	void make(size_t r, size_t c, real_t const *p);

	real_t &at(size_t r, size_t c)
	{
		return d->vals[d->cols * r + c];
	}

	real_t at(size_t r, size_t c) const
	{
		return d->vals[d->cols * r + c];
	}

	static real_t sigmoid(real_t v)
	{
		return 1 / (1 + exp(-v));
	}

	Matrix transpose() const;
	Matrix dot(Matrix const &other) const;
	Matrix add(Matrix const &other) const;
	Matrix sub(Matrix const &other) const;
	Matrix mul(Matrix const &other) const;
	Matrix mul(real_t t) const;
	Matrix div(real_t t) const;
	Matrix sum() const;
	Matrix sigmoid() const;
	Matrix sigmoid_grad() const;
	Matrix softmax() const;

	void add_rows(Matrix const &other);

	Matrix cross_entropy_error(Matrix const &t) const
	{
		if (t.d->rows != d->rows) return {};
		if (t.d->cols != d->cols) return {};
		Matrix out;
		out.make(rows(), 1);
		const double delta = 1e-7;
		for (size_t r = 0; r < d->rows; r++) {
			real_t e = 0;
			for (size_t c = 0; c < d->cols; c++) {
				e += t.at(r, c) * log(at(r, c) + delta);
			}
			out.at(r, 0) = e;
		}
		return out;
	}
};

namespace mnist {

class DataSet {
private:

	struct Data {
		size_t count = 0;
		int rows = 0;
		int cols = 0;
		std::vector<uint8_t> labels;
		std::vector<std::vector<uint8_t>> images;
	} data;

public:
	bool load(char const *labels_path, char const *images_path);

	size_t size() const
	{
		return data.count;
	}

	bool image_to_matrix(int index, Matrix *out) const;
	void label_to_matrix(int index, Matrix *out) const;
	int label(int index) const;
};


} // namespace mnist

#endif // MNIST_H