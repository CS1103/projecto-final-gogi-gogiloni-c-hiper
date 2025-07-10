//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_loss.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include <vector>
#include <memory>
#include <cassert>
#include <fstream>
#include <string>

namespace utec {
namespace neural_network {

template<typename T>  
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers;

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers.emplace_back(std::move(layer));
    }

    template<
        template <typename...> class LossType, 
        template <typename...> class OptimizerType = SGD
    >
    void train(
        const algebra::Tensor<T, 2>& X,
        const algebra::Tensor<T, 2>& Y,
        const size_t epochs,
        const size_t batch_size,
        T learning_rate
    ) {
        const size_t total = X.shape()[0];
        assert(total == Y.shape()[0]);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < total; i += batch_size) {
                size_t current_batch = std::min(batch_size, total - i);

                algebra::Tensor<T, 2> x_batch(current_batch, X.shape()[1]);
                algebra::Tensor<T, 2> y_batch(current_batch, Y.shape()[1]);

                for (size_t j = 0; j < current_batch; ++j) {
                    for (size_t k = 0; k < X.shape()[1]; ++k)
                        x_batch(j, k) = X(i + j, k);
                    for (size_t k = 0; k < Y.shape()[1]; ++k)
                        y_batch(j, k) = Y(i + j, k);
                }

                auto predictions = x_batch;
                for (auto& layer : layers)
                    predictions = layer->forward(predictions);

                LossType<T> loss_fn(predictions, y_batch);
                auto loss_grad = loss_fn.loss_gradient();

                for (auto it = layers.rbegin(); it != layers.rend(); ++it)
                    loss_grad = (*it)->backward(loss_grad);

                OptimizerType<T> opt(learning_rate);

                for (auto& layer : layers)
                    layer->update_params(opt);
            }
        }
    }

    algebra::Tensor<T, 2> predict(const algebra::Tensor<T, 2>& X) {
        auto predictions = X;
        for (auto& layer : layers)
            predictions = layer->forward(predictions);
        return predictions;
    }

    void save(const std::string& path) const {
        std::ofstream out(path);
        for (const auto& layer : layers) {
            if (auto dense = dynamic_cast<Dense<T>*>(layer.get())) {
                out << "Dense\n";
                dense->save(out);
            } else if (dynamic_cast<ReLU<T>*>(layer.get())) {
                out << "ReLU\n";
            } else if (dynamic_cast<Sigmoid<T>*>(layer.get())) {
                out << "Sigmoid\n";
            }
        }
        out.close();
    }

    void load(const std::string& path) {
        std::ifstream in(path);
        std::string type;
        layers.clear();

        while (in >> type) {
            if (type == "Dense") {
                auto dense = std::make_unique<Dense<T>>(
                    0, 0, 
                    [](auto&){}, 
                    [](auto&){}
                );
                dense->load(in);
                layers.push_back(std::move(dense));
            } else if (type == "ReLU") {
                layers.push_back(std::make_unique<ReLU<T>>());
            } else if (type == "Sigmoid") {
                layers.push_back(std::make_unique<Sigmoid<T>>());
            }
        }
        in.close();
    }

    const std::vector<std::unique_ptr<ILayer<T>>>& dlayers() const {
        return layers;
    }
};

} // namespace neural_network
} // namespace utec

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
