#include "../utec/neural_network/data/mnist_loader.h"

using namespace std;
int jijijia() {
    MNISTLoader loader;
    
    if (!loader.loadTrainData("mnist/mnist_train.csv")) {
        cerr << "Error cargando datos de entrenamiento" << endl;
        return -1;
    }
    
    if (!loader.loadTestData("mnist/mnist_test.csv")) {
        cerr << "Error cargando datos de test" << endl;
        return -1;
    }
    
    loader.printDatasetInfo();
    
    cout << "\n=== Muestra 0 ===" << endl;
    loader.printSample(0);
    
    cout << "\n=== Probando batch ===" << endl;
    auto [batch_images, batch_labels] = loader.getBatch(5);
    
    cout << "Batch obtenido:" << endl;
    for (int i = 0; i < batch_labels.size(); i++) {
        cout << "Imagen " << i << ": label = " << batch_labels[i] 
                  << ", tamanio = " << batch_images[i].size() << endl;
    }
    
    return 0;
}