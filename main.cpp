#include <iostream>
#include <vector>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

template<typename Derived>
void printMatrix(const MatrixBase<Derived>& mat){
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            cout << mat(i, j) << " ";
        }
        cout << endl;
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

MatrixXf sigmoid(const MatrixXf& mat) {
    MatrixXf result(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            result(i, j) = sigmoid(mat(i, j));
        }
    }
    return result;
}

void sigmoidInPlace(MatrixXf& mat){
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            mat(i, j) = sigmoid(mat(i, j));
        }
    }
}

class NeuralNetwork {
    public:
        vector<MatrixXf> weights;
        NeuralNetwork(vector<int> shape) {
            for (int i = 0; i < shape.size() - 1; ++i) {
                MatrixXf weight = MatrixXf::Random(shape[i + 1], shape[i]);
                weights.push_back(weight);
            }
        }

        void printWeights() {
            for (int i = 0; i < weights.size(); ++i) {
                cout << "Weights " << i << ":" << endl;
                printMatrix(weights[i]);
                cout << "---------" << endl;
            }
        }

        MatrixXf query(const MatrixXf& input) {
            MatrixXf output = input;
            for (int i = 0; i < weights.size(); ++i) {
                printMatrix(weights[i]);
                cout << "---------" << endl;
                printMatrix(output);
                cout << "---------" << endl;
                output = (weights[i] * output);
                //sigmoidInPlace(output);
                output = sigmoid(output); // Uncomment this line if you want to use the non-in-place version
            }
            cout << "Output: " << endl;
            printMatrix(output);
            return output;
        }


};

int main() {
    Matrix <float, 3, 3> mat;
    Matrix <float, 3, 1> i;
    NeuralNetwork nn({3, 3 , 1});
    nn.printWeights();
    i << 0.5, 0.5, 0.5;
    nn.query(i);
    //TODO add training function
    //TODO get training from MNIST dataset
    cout << "Hello World!" << endl;
    return 0;
}