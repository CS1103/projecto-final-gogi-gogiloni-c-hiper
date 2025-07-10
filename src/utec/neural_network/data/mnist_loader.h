#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric> 
#include "../../algebra/tensor.h"

using namespace std;
class MNISTLoader {
private:
    vector<vector<float>> train_images;
    vector<int> train_labels;
    vector<vector<float>> test_images;
    vector<int> test_labels;

    int current_batch_index;
    mt19937 rng;

public:
    MNISTLoader() : current_batch_index(0), rng(42) {}

    bool loadTrainData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: No se pudo abrir " << filename << endl;
            return false;
        }

        string line;
        getline(file, line);

        while (getline(file, line)) {
            if (line.empty()) continue;

            int label = parseLabelFromRow(line);
            vector<float> image = parseImageRow(line);

            if (image.size() == 784) {
                train_labels.push_back(label);
                train_images.push_back(image);
            }
        }

        file.close();
        normalizeData();

        cout << "Cargados " << train_images.size() << " ejemplos de entrenamiento" << endl;
        return true;
    }

    bool loadTestData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: No se pudo abrir " << filename << endl;
            return false;
        }

        string line;
        getline(file, line); 

        while (getline(file, line)) {
            if (line.empty()) continue;

            int label = parseLabelFromRow(line);
            vector<float> image = parseImageRow(line);

            if (image.size() == 784) {
                test_labels.push_back(label);
                test_images.push_back(image);
            }
        }

        file.close();
        cout << "Cargados " << test_images.size() << " ejemplos de test" << endl;
        return true;
    }

    pair<vector<vector<float>>, vector<int>> getTrainData() {
        return {train_images, train_labels};
    }

    pair<vector<vector<float>>, vector<int>> getTestData() {
        return {test_images, test_labels};
    }

    pair<vector<vector<float>>, vector<int>> getBatch(int batch_size) {
        vector<vector<float>> batch_images;
        vector<int> batch_labels;

        for (size_t i = 0; i < static_cast<size_t>(batch_size) && current_batch_index < train_images.size(); i++) {
            batch_images.push_back(train_images[current_batch_index]);
            batch_labels.push_back(train_labels[current_batch_index]);
            current_batch_index++;
        }

        if (current_batch_index >= train_images.size()) {
            current_batch_index = 0;
            shuffleTrainData();
        }

        return {batch_images, batch_labels};
    }

    void shuffleTrainData() {
        vector<int> indices(train_images.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);

        vector<vector<float>> shuffled_images;
        vector<int> shuffled_labels;

        for (int idx : indices) {
            shuffled_images.push_back(train_images[idx]);
            shuffled_labels.push_back(train_labels[idx]);
        }

        train_images = shuffled_images;
        train_labels = shuffled_labels;
    }

    void normalizeData() {
        for (auto& image : train_images) {
            for (float& pixel : image) {
                pixel /= 255.0f;
            }
        }

        for (auto& image : test_images) {
            for (float& pixel : image) {
                pixel /= 255.0f;
            }
        }
    }

    void printSample(int index, bool is_train = true) {
        const auto& images = is_train ? train_images : test_images;
        const auto& labels = is_train ? train_labels : test_labels;

        if (index >= static_cast<int>(images.size())) {
            cout << "Indice fuera de rango" << endl;
            return;
        }

        cout << "Etiqueta: " << labels[index] << endl;
        cout << "Imagen (28x28):" << endl;

        for (size_t i = 0; i < 28; i++) {
            for (size_t j = 0; j < 28; j++) {
                float pixel = images[index][i * 28 + j];
                if (pixel > 0.5) cout << "##";
                else if (pixel > 0.25) cout << "..";
                else cout << "  ";
            }
            cout << endl;
        }
    }

    void printDatasetInfo() {
        cout << "  MNIST Dataset Info" << endl;
        cout << "Entrenamiento: " << train_images.size() << " imagenes" << endl;
        cout << "Test: " << test_images.size() << " imagenes" << endl;
        cout << "Dimension de imagen: 784 (28x28)" << endl;
        cout << "Clases: 0-9 (10 dígitos)" << endl;
    }

    int getTrainSize() const { return train_images.size(); }
    int getTestSize() const { return test_images.size(); }

private:
    vector<string> split(const string& str, char delimiter) {
        vector<string> tokens;
        stringstream ss(str);
        string token;

        while (getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }

    vector<float> parseImageRow(const string& row) {
        vector<string> tokens = split(row, ',');
        vector<float> image;

        for (size_t i = 1; i < tokens.size(); i++) {
            image.push_back(stof(tokens[i]));
        }

        return image;
    }

    int parseLabelFromRow(const string& row) {
        vector<string> tokens = split(row, ',');
        return stoi(tokens[0]);
    }
};
