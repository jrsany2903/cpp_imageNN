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
MatrixXf oneMat(int rows){
    MatrixXf mat(rows, 1);
    for (int i = 0; i < rows; ++i) {
        mat(i, 0) = 1.0f;
    }
    return mat;
}

class NeuralNetwork {
    public:
        vector<MatrixXf> weights;
        vector<int> shape;
        float learningRate; // Learning rate for weight updates
        NeuralNetwork(vector<int> shape, float learningRate = 0.01f) {
            this->shape = shape;
            this->learningRate = learningRate;
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
        void trainOne(const MatrixXf& input, const MatrixXf& expected) {
            vector<MatrixXf> outputs = this->detQuery(input);
            MatrixXf output, prev, delta;
            output = outputs.back(); // Get the last output
            MatrixXf error = expected - output;
            for (int i = weights.size() - 1; i >= 0; --i) {
                //delta = lr * e * output * (1 - output) * input^T
                output = outputs[i + 1]; // Get the output of the current layer
                prev = outputs[i]; // Get the input of the current layer
                delta = this->learningRate * (error.cwiseProduct(output).cwiseProduct(oneMat(output.rows()) - output)) * prev.transpose();// Code Breaks here
                weights[i] += delta; // Update weights
                error = weights[i].transpose() * error; // Backpropagation error
            }
        }

        void train(vector<MatrixXf> data, vector<MatrixXf> Expected){
            for(int i = 0; i < data.size(); i++){
                MatrixXf input = data[i];
                MatrixXf expected = Expected[i];
                this->trainOne(input, expected);
            }
        }

        vector<MatrixXf> detQuery(const MatrixXf& input) {
            vector<MatrixXf> outputs;
            MatrixXf output = input;
            outputs.push_back(output);
            for (int i = 0; i < weights.size(); ++i) {
                output = (weights[i] * output);
                output = sigmoid(output); 
                outputs.push_back(output);
            }
            return outputs;
        }

        MatrixXf query(const MatrixXf& input) {
            return this->detQuery(input).back(); // Get the last output
        }


};

int main() {
    Matrix <float, 3, 3> mat;
    Matrix <float, 3, 1> i;
    NeuralNetwork nn({784, 28 , 28, 10});
    
    nn.printWeights();
    printMatrix(nn.query(i));
    i << 0.5, 0.5, 0.5;
    for (int k = 0; k < 2000; ++k) {
        nn.trainOne(i, oneMat(3));
    }

    nn.printWeights();
    printMatrix(nn.query(i));
    //TODO add training function
    //TODO get training from MNIST dataset
    cout << "Hello World!" << endl;
    return 0;
}