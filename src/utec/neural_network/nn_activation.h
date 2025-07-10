//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec
{
    namespace algebra
    {
        template<typename T, size_t DIMS, typename Func>
        Tensor<T, DIMS> apply(const Tensor<T, DIMS>& t, Func func) {
            auto shape = t.shape();
            Tensor<T, DIMS> result(shape[0], shape[1]);
            for (size_t i = 0; i < shape[0]; ++i){
                for (size_t j = 0; j < shape[1]; ++j){
                    result(i, j) = func(t(i, j));
                }
            }
            return result;
        }

    } // namespace algebra
    
} // namespace utec



namespace utec
{
    namespace neural_network
    {
        template<typename T>
        class ReLU final : public ILayer<T> {
        private:
            algebra::Tensor<T, 2> input;

        public:
            ReLU() : input(1, 1) {}

            algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2>& z) override {
                input = z;
                auto shape = z.shape();
                algebra::Tensor<T, 2> result(shape[0], shape[1]);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                    result(i, j) = std::max(T(0), z(i, j));
                    }
                }
                return result;
            }

            algebra::Tensor<T, 2> backward(const algebra::Tensor<T, 2>& g) override {
                auto shape = g.shape();
                algebra::Tensor<T, 2> grand(shape[0], shape[1]);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                    grand(i, j) = input(i, j) > T(0) ? g(i, j) : T(0);
                    }
                }
                return grand;
            }
        };
        template<typename T>
        class Sigmoid final : public ILayer<T> {
        private:
            algebra::Tensor<T, 2> output;
            static T sigmoid(T x) {
                return T(1) / (T(1) + std::exp(-x));
            }

        public:
            Sigmoid() : output(1, 1) {}

            algebra::Tensor<T, 2> forward(const algebra::Tensor<T, 2>& z) override {
                auto shape = z.shape();
                output = algebra::Tensor<T, 2>(shape[0], shape[1]);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        output(i, j) = sigmoid(z(i, j));
                    }
                }
                return output;
            }

            algebra::Tensor<T, 2> backward(const algebra::Tensor<T, 2>& g) override {
            auto shape = g.shape();
                algebra::Tensor<T, 2> grand(shape[0], shape[1]);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        T sig = output(i, j);
                        grand(i, j) = g(i, j) * sig * (T(1) - sig);
                    }
                }
                return grand;
            }
        };
    } // namespace neural_network
    
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
