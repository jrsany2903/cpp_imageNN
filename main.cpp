#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>



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

int getIndexOfMax(MatrixXf& mat) {
    int index = 0;
    float maxVal = mat(0, 0);
    for (int i = 1; i < mat.rows(); ++i) {
        if (mat(i, 0) > maxVal) {
            maxVal = mat(i, 0);
            index = i;
        }
    }
    return index;
}

int main(int argc , char* argv[]) {

    NeuralNetwork nn({784, 28 , 28, 10});

    ifstream trainingdata("Original Dataset/train-images.idx3-ubyte", ios::binary);
    ifstream traininglabels("Original Dataset/train-labels.idx1-ubyte", ios::binary);
    trainingdata.seekg(16, ios::beg);
    traininglabels.seekg(8, ios::beg);

    unsigned char data[784];
    unsigned char label[1];
    Matrix <float, 784, 1> image; 
    Matrix <float, 10, 1> labelmat;

    for (int dataset = 0; dataset < 60000; ++dataset) {
        image.setZero();
        labelmat.setZero();
        trainingdata.read((char*)data, 784);
        traininglabels.read((char*)label, 1);   
        for (int i = 0; i < 784; ++i) {
            image(i, 0) = (((float)((int)data[i]) / 255.0f) * 0.99f) + 0.01f; // Normalize to [0.01, 0.99]
        }
        nn.learningRate = max(0.05f * (dataset / 60000.0f), 0.01f); // Decrease learning rate over time
        labelmat((int)label[0], 0) = 0.99f; // Set the label to 0.99 for the correct class
        nn.trainOne(image, labelmat); 
        cout << "finished training on image " << dataset << endl;
    }
    cout << "Finished training on 60000 images" << endl;
    trainingdata.close();
    traininglabels.close();

    //TODO add testing data
    //TODO add saving and loading nn from bin files
    //TODO create trial for different learning rates and NN shapes


    //Testing
    ifstream testingdata("Original Dataset/t10k-images.idx3-ubyte", ios::binary);
    ifstream testinglabels("Original Dataset/t10k-labels.idx1-ubyte", ios::binary);
    testingdata.seekg(16, ios::beg);
    testinglabels.seekg(8, ios::beg);
    int count = 0;

    MatrixXf output;

    for (int dataset = 0; dataset < 10000; ++dataset) {
        image.setZero();
        labelmat.setZero();
        testingdata.read((char*)data, 784);
        testinglabels.read((char*)label, 1);   
        for (int i = 0; i < 784; ++i) {
            image(i, 0) = (((float)((int)data[i]) / 255.0f) * 0.99f) + 0.01f; // Normalize to [0.01, 0.99]
        }
        labelmat((int)label[0], 0) = 0.99f; // Set the label to 0.99 for the correct class
        cout << "Testing on image " << dataset << endl;
        output = nn.query(image); // Get the output of the neural network
        if (getIndexOfMax(output) == (int)label[0]) { 
            count++;
        }
    }

    testingdata.close();
    testinglabels.close();
    cout << "Finished testing on 10000 images" << endl;
    cout << count << " images were classified correctly" << endl;


    cout << "Program Terminated..." << endl;
    return 0;
}