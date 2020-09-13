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
        size_t rows = 0;
        size_t cols = 0;
        std::vector<real_t> vals;
    };
    std::shared_ptr<Data> d;

public:

    // コンストラクタ
    // mnist.cppにかかれる
    Matrix();
    Matrix(Matrix const &r);

    // ここは何？
    // = でコンストラクタできるようにしてる？
    void operator = (Matrix const &r)
    {
        *d = *r.d;
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

    // こういう宣言には、終わりにセミコロンいるんだな
    Matrix transpose() const;
    Matrix dot(Matrix const &othr) const;
    Matrix add(Matrix const &othr) const;
    Matrix sub(Matrix const &othr) const;
    Matrix mul(Matrix const &othr) const;
    Matrix mul(real_t t) const;
    Matrix div(real_t t) const;
    Matrix sum() const;
    Matrix sigmoid() const;
    Matrix sigmoid_grad() const;
    Matrix softmax() const;

    void add_rows(Matrix const &other);
};

class Layer {
public:
    Matrix weight;
    Matrix bias;

    // ここのinputとoutputはなんでそのまま渡していいんだろう？
    void reset(int input, int output)
    {
        weight.make(input, output);
        bias.make(1, output);
    }
};

namespace mnist {
class Dataset {
    private:
        struct Data {
            size_t count = 0;
            int rows = 0;
            int cols = 0;
            std::vector<uint8_t> labels;
            std::vector<std::vector<uint8_t>> images;
        } data;

    public:
        // 中身はtest_mnist.cpp
        bool load(char const *labels_path, char const *images_path);

        // 値を書き換えない関数だからconstつける
        // こうやって値を取り出すだけならhに書いていいのかな
        size_t size() const
        {
            return data.count;
        }

        // 中身はtest_mnist.cpp
        bool image_to_matrix(int index, Matrix *out) const;
        void label_to_matrix(int index, Matrix *out) const;
        int label(int index) const;
};
}

#endif // MNIST_H