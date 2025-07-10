//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#include "nn_interfaces.h"
#include "../algebra/tensor.h"

namespace utec {
namespace neural_network {

template<typename T>
class Dense final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> weights;
    utec::algebra::Tensor<T, 2> biases;
    utec::algebra::Tensor<T, 2> input;
    utec::algebra::Tensor<T, 2> grad_w;
    utec::algebra::Tensor<T, 2> grad_b;

public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun) :
        weights(in_f, out_f),
        biases(1, out_f),
        input(1, in_f),
        grad_w(in_f, out_f),
        grad_b(1, out_f) {
        init_w_fun(weights);
        init_b_fun(biases);
    }

    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
        input = x;
        auto out = utec::algebra::matrix_product(x, weights);
        return out + biases;
    }

    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& dZ) override {
        grad_w = utec::algebra::matrix_product(utec::algebra::transpose_2d(input), dZ);
        grad_b = utec::algebra::Tensor<T, 2>(1, dZ.shape()[1]);
        for (size_t j = 0; j < dZ.shape()[1]; ++j) {
            T sum = T(0);
            for (size_t i = 0; i < dZ.shape()[0]; ++i) {
                sum += dZ(i, j);
            }
            grad_b(0, j) = sum;
        }
        return utec::algebra::matrix_product(dZ, utec::algebra::transpose_2d(weights));
    }

    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights, grad_w);
        optimizer.update(biases, grad_b);
        optimizer.step();
    }

    void save(std::ostream& out) const {
        size_t rows = weights.shape()[0];
        size_t cols = weights.shape()[1];
        out << rows << " " << cols << "\n";
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                out << weights(i,j) << " ";
        out << "\n";

        rows = biases.shape()[0];
        cols = biases.shape()[1];
        out << rows << " " << cols << "\n";
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                out << biases(i,j) << " ";
        out << "\n";
    }

    void load(std::istream& in) {
        size_t rows, cols;

        in >> rows >> cols;
        std::array<size_t, 2> w_shape = {rows, cols};
        weights = utec::algebra::Tensor<T, 2>(w_shape);

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                in >> weights(i,j);

        in >> rows >> cols;
        std::array<size_t, 2> b_shape = {rows, cols};
        biases = utec::algebra::Tensor<T, 2>(b_shape);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                in >> biases(i,j);
    }
};

} // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
