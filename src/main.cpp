#include <iostream>
#include <memory>
#include <random>
#include <chrono>

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
float random_weight() {
    static mt19937 gen(random_device{}());
    static uniform_real_distribution<float> dist(-0.1f, 0.1f);
    return dist(gen);
}

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
    MNISTLoader loader;

    cout << "Cargando datos..." << endl;
    loader.loadTrainData("mnist/mnist_train.csv");
    loader.loadTestData("mnist/mnist_test.csv");

    auto [train_images, train_labels] = loader.getTrainData();
    auto [test_images, test_labels] = loader.getTestData();

    auto X_train = vector2D_to_tensor(train_images);
    auto Y_train = labels_to_onehot(train_labels, 10);

    auto X_test = vector2D_to_tensor(test_images);
    auto Y_test = labels_to_onehot(test_labels, 10);

    cout << "Tamanio entrenamiento: " << train_images.size() << endl;
    cout << "Tamanio prueba: " << test_images.size() << endl;

    NeuralNetwork<float> nn;

    nn.add_layer(make_unique<Dense<float>>(
        784, 128,
        [](auto& w){ for (auto& val : w) val = random_weight(); },
        [](auto& b){ b.fill(0.0f); }
    ));
    nn.add_layer(make_unique<ReLU<float>>());

    nn.add_layer(make_unique<Dense<float>>(
        128, 64,
        [](auto& w){ for (auto& val : w) val = random_weight(); },
        [](auto& b){ b.fill(0.0f); }
    ));
    nn.add_layer(make_unique<ReLU<float>>());

    nn.add_layer(make_unique<Dense<float>>(
        64, 10,
        [](auto& w){ for (auto& val : w) val = random_weight(); },
        [](auto& b){ b.fill(0.0f); }
    ));
    nn.add_layer(make_unique<Sigmoid<float>>());

    cout << "Entrenando red..." << endl;

    auto start = chrono::high_resolution_clock::now();


    size_t epochs = 100;
    size_t batch_size = 64;
    float learning_rate = 0.01f;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        nn.train<BCELoss, Adam>(X_train, Y_train, 1, batch_size, learning_rate);


        if ((epoch+1) % 10 == 0 || epoch == epochs-1) {
            cout << "Epoca " << (epoch+1) << "/" << epochs << " completada\n";
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Entrenamiento terminado en " << elapsed.count() << " segundos\n";

    cout << "Evaluando en conjunto de prueba..." << endl;
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


    ofstream out("modelo.nn");
    for (const auto& layer : nn.dlayers()) {
        if (auto dense = dynamic_cast<Dense<float>*>(layer.get())) {
            out << "Dense\n";
            dense->save(out);
        } else if (dynamic_cast<ReLU<float>*>(layer.get())) {
            out << "ReLU\n";
        } else if (dynamic_cast<Sigmoid<float>*>(layer.get())) {
            out << "Sigmoid\n";
        }
    }
    out.close();
    cout << "Modelo guardado en modelo.nn\n";
    return 0;
}
