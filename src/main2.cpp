#include <iostream>
#include <memory>
#include <vector>
#include "utec/algebra/tensor.h"
#include "utec/neural_network/neural_network.h"
#include "utec/neural_network/nn_dense.h"
#include "utec/neural_network/nn_activation.h"
#include "utec/neural_network/nn_loss.h"
#include "utec/neural_network/nn_optimizer.h"
#include "utec/neural_network/data/mnist_loader.h"

using namespace utec::algebra;
using namespace utec::neural_network;
using namespace std;

Tensor<float, 2> vector2D_to_tensor(const vector<vector<float>>& data) {
    size_t rows = data.size();
    size_t cols = data[0].size();
    Tensor<float, 2> tensor(array<size_t, 2>{rows, cols});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            tensor(i, j) = data[i][j];
    return tensor;
}

Tensor<float, 2> labels_to_onehot(const vector<int>& labels, int num_classes) {
    size_t rows = labels.size();
    Tensor<float, 2> tensor(array<size_t, 2>{rows, static_cast<size_t>(num_classes)});
    tensor.fill(0.0f);
    for (size_t i = 0; i < rows; ++i)
        tensor(i, labels[i]) = 1.0f;
    return tensor;
}

int main() {
    cout << "Cargando datos de prueba..." << endl;

    MNISTLoader loader;
    loader.loadTestData("mnist/mnist_test.csv");
    auto [test_images, test_labels] = loader.getTestData();

    auto X_test = vector2D_to_tensor(test_images);
    auto Y_test = labels_to_onehot(test_labels, 10);

    cout << "Tamanio prueba: " << test_images.size() << endl;

    NeuralNetwork<float> nn;

    nn.load("modelo.nn");

    cout << "Evaluando modelo cargado..." << endl;

    auto Y_pred = nn.predict(X_test);

    int correct = 0;
    for (size_t i = 0; i < Y_pred.shape()[0]; ++i) {
        int pred_label = 0;
        float max_val = Y_pred(i, 0);
        for (int j = 1; j < 10; ++j) {
            if (Y_pred(i, j) > max_val) {
                max_val = Y_pred(i, j);
                pred_label = j;
            }
        }
        if (pred_label == test_labels[i])
            ++correct;
    }

    float accuracy = static_cast<float>(correct) / test_labels.size() * 100.0f;
    cout << "Precision en prueba: " << accuracy << " %" << endl;

    return 0;
}
