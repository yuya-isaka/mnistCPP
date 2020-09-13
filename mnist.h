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
        site_t cols = 0;
        std::vector<real_t> vals
    }
}

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