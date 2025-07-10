//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#include "nn_interfaces.h"
#include <vector>
#include <cmath>

namespace utec {
namespace neural_network {

template<typename T>
class SGD final : public IOptimizer<T> {
private:
    T l_r;

public:
    explicit SGD(T learning_rate = 0.01) : l_r(learning_rate) {}

    void update(utec::algebra::Tensor<T, 2>& params,
                const utec::algebra::Tensor<T, 2>& grads) override {
        for (size_t i = 0; i < params.size(); ++i)
            params.begin()[i] -= l_r * grads.cbegin()[i];
    }
};

template<typename T>
class Adam final : public IOptimizer<T> {
private:
    T lr_, beta1_, beta2_, epsilon_;
    size_t t_ = 0;
    std::vector<utec::algebra::Tensor<T, 2>> t1, t2;

public:
    explicit Adam(T lr = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

    void update(utec::algebra::Tensor<T, 2>& params,
                const utec::algebra::Tensor<T, 2>& grads) override {
        if (t1.empty()) {
            t1.emplace_back(grads.shape()[0], grads.shape()[1]);
            t2.emplace_back(grads.shape()[0], grads.shape()[1]);
            t1[0].fill(T(0));
            t2[0].fill(T(0));
        }

        t_++;
        auto& T1 = t1[0];
        auto& T2 = t2[0];

        for (size_t i = 0; i < grads.size(); ++i) {
            T1.begin()[i] = beta1_ * T1.begin()[i] + (1 - beta1_) * grads.cbegin()[i];
            T2.begin()[i] = beta2_ * T2.begin()[i] + (1 - beta2_) * grads.cbegin()[i] * grads.cbegin()[i];

            T t1_hat = T1.begin()[i] / (1 - std::pow(beta1_, T(t_)));
            T t2_hat = T2.begin()[i] / (1 - std::pow(beta2_, T(t_)));

            params.begin()[i] -= lr_ * t1_hat / (std::sqrt(t2_hat) + epsilon_);
        }
    }

    void step() override {}
};

} // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
