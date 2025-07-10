//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#include "nn_interfaces.h"
#include <cmath>
#include <algorithm>

namespace utec
{
    namespace neural_network
    {   
        template<typename T>
        class MSELoss final : public ILoss<T, 2>
        {
        private:
            algebra::Tensor<T, 2> y_pred;
            algebra::Tensor<T, 2> y_true;
            T loss_ = T(0);
        public:
            MSELoss(const algebra::Tensor<T, 2>& y_p, const algebra::Tensor<T, 2>& y_t) : y_pred(y_p),y_true(y_t){
                auto shape = y_pred.shape();
                T sum = T(0);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        T diff = y_pred(i, j) - y_true(i, j);
                        sum += diff * diff;
                    }
                }
                loss_ = sum / (shape[0] * shape[1]);
            }

            T loss() const override {
                return loss_;
            }

             algebra::Tensor<T, 2> loss_gradient() const override {
                auto shape = y_pred.shape();
                algebra::Tensor<T, 2> grad(shape[0], shape[1]);
                T coef = T(2) / (shape[0] * shape[1]);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        grad(i, j) = coef * (y_pred(i, j) - y_true(i, j));
                    }
                }
                return grad;
            }    
        };
        
        template<typename T>
        class BCELoss final : public ILoss<T, 2> {
        private:
            algebra::Tensor<T, 2> y_pred;
            algebra::Tensor<T, 2> y_true;
            T loss_ = T(0);
            const T e = 1e-7;

        public:
            BCELoss(const algebra::Tensor<T, 2>& y_p, const algebra::Tensor<T, 2>& y_t): y_pred(y_p), y_true(y_t) {
                auto shape = y_pred.shape();
                T sum = T(0);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        T y = y_true(i, j);
                        T p = std::clamp(y_pred(i, j), e, T(1) - e);
                        sum += -y * std::log(p) - (1 - y) * std::log(1 - p);
                    }
                }
                loss_ = sum / (shape[0] * shape[1]);
            }

            T loss() const override {
                return loss_;
            }

            algebra::Tensor<T, 2> loss_gradient() const override {
                auto shape = y_pred.shape();
                algebra::Tensor<T, 2> grad(shape[0], shape[1]);
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j) {
                        T y = y_true(i, j);
                        T p = std::clamp(y_pred(i, j), e, T(1) - e);
                        grad(i, j) = (p - y) / (p * (1 - p) * shape[0] * shape[1]);
                    }
                }
                return grad;
            }
        };

    } // namespace neural_network
    
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
