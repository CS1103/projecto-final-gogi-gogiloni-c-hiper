//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#include "../algebra/tensor.h"

namespace utec::neural_network {

  // Interfaz del optimizador (SGD o Adam)
  template<typename T>
  struct IOptimizer {
    virtual ~IOptimizer() = default;
    virtual void update(utec::algebra::Tensor<T,2>& params, 
                        const utec::algebra::Tensor<T,2>& gradients) = 0;
    virtual void step() {}
  };

  // Interfaz de las capas (Dense y los diferentes tipos de activación)
  template<typename T>
  struct ILayer {
    virtual ~ILayer() = default;

    virtual utec::algebra::Tensor<T,2> forward(
        const utec::algebra::Tensor<T,2>& x) = 0;

    virtual utec::algebra::Tensor<T,2> backward(
        const utec::algebra::Tensor<T,2>& gradients) = 0;

    // Se utiliza para actualizar los parameters a través del optimizador
    // Se puede llamar tanto el método update y step si es requerido
    virtual void update_params(IOptimizer<T>& optimizer) {}
  };

  // Interfaz de las pérdidas (MSE o BCE)
  template<typename T, size_t DIMS>
  struct ILoss {
    virtual ~ILoss() = default;

    virtual T loss() const = 0;

    virtual utec::algebra::Tensor<T,DIMS> loss_gradient() const = 0;
  };

} // namespace utec::neural_network

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
