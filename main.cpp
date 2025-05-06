#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <fstream>
#include <random>

using namespace std;
using namespace Eigen;

template <typename Derived>
void printMatrix(const MatrixBase<Derived> &mat)
{
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            cout << mat(i, j) << " ";
        }
        cout << endl;
    }
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

MatrixXf sigmoid(const MatrixXf &mat)
{
    MatrixXf result(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            result(i, j) = sigmoid(mat(i, j));
        }
    }
    return result;
}

void sigmoidInPlace(MatrixXf &mat)
{
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            mat(i, j) = sigmoid(mat(i, j));
        }
    }
}

MatrixXf oneMat(int rows)
{
    MatrixXf mat(rows, 1);
    for (int i = 0; i < rows; ++i)
    {
        mat(i, 0) = 1.0f;
    }
    return mat;
}

float sumRow(MatrixXf &mat, int row)
{
    float sum = 0.0f;
    for (int i = 0; i < mat.cols(); ++i)
    {
        sum += mat(row, i);
    }
    return sum;
}

int getIndexOfMax(MatrixXf &mat)
{
    int index = 0;
    float maxVal = mat(0, 0);
    for (int i = 1; i < mat.rows(); ++i)
    {
        if (mat(i, 0) > maxVal)
        {
            maxVal = mat(i, 0);
            index = i;
        }
    }
    return index;
}

MatrixXf fnInPlace(MatrixXf &mat, function<float(float)> fn)
{
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            mat(i, j) = fn(mat(i, j));
        }
    }
    return mat;
}

class NeuralNetwork
{
private:
    MatrixXf NormalDist(float mean, float sigma, int rows, int cols)
    {
        MatrixXf mat(rows, cols);
        normal_distribution<float> nd(mean, sigma);
        random_device rd;
        mt19937 gen(rd());
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                mat(i, j) = nd(gen); // Generate random number from normal distribution
            }
        }
        return mat;
    }
    static float sigmoidf(float x)
    {
        return 1.0f / (1.0f + exp(-x));
    }
    static float sigmoidDerivative(float x)
    {
        return sigmoidf(x) * (1.0f - sigmoidf(x));
    }
    static float relu(float x)
    {
        return (x > 0) ? x : 0;
    }

    static float reluDerivative(float x)
    {
        return (x > 0) ? 1 : 0;
    }

public:
    unordered_map<string, array<function<float(float)>, 2>> activationFunctionsDict = {
        {"sigmoid", array<function<float(float)>, 2>{sigmoidf, sigmoidDerivative}},
        {"relu", array<function<float(float)>, 2>{relu, reluDerivative}}};
    vector<MatrixXf> weights;
    vector<MatrixXf> biases;
    vector<int> shape;
    vector<string> activationFunctionsStr; // pass vector<nullaryexpr> to store activation functions for each layer
    // pass vector<nullaryexpr> to store activation functions for each layer
    // also store the derivative of the activation function for each layer
    // vector<function<float(float)>> activationFunctions; chatgpt suggestion
    float learningRate; // Learning rate for weight updates
    NeuralNetwork(vector<int> shape, float learningRate, vector<string> actvfns = {"sigmoid"})
    {
        MatrixXf weight, bias;
        this->shape = shape;
        this->learningRate = learningRate;
        for (int i = 0; i < shape.size() - 1; ++i)
        {
            // use MatrixXf::NullaryEpr() to initialise weights and biases to normal distribution with sigma 1 / root(shape[i + 1])
            // std::random_device rd;
            // std::mt19937 gen(rd());
            // std::normal_distribution<float> dist(0.0f, 1.0f); // Mean = 0, StdDev = 1
            // MatrixXf weight = MatrixXf::NullaryExpr(shape[i + 1], shape[i], [&]() { return dist(gen); });

            // weight = MatrixXf::Random(shape[i + 1], shape[i]);
            weight = NormalDist(0.0f, 1.0f / sqrt(shape[i + 1]), shape[i + 1], shape[i]); // Initialize weights with normal distribution
            // bias = MatrixXf::Random(shape[i + 1], 1);
            bias = NormalDist(0.0f, 1.0f / sqrt(shape[i + 1]), shape[i + 1], 1); // Initialize biases with normal distribution
            weights.push_back(weight);
            biases.push_back(bias);
            activationFunctionsStr.push_back(actvfns[i % actvfns.size()]); // Store the activation function for the layer
        }
    }
    NeuralNetwork(string path)
    {
        // load in the neural network from a file
        ifstream file(path, ios::binary);
        if (!file)
        {
            cerr << "Error opening file for reading: " << path << endl;
            return;
        }

        int actfnSize, stringsize;
        string actfn;

        file.read((char *)&this->learningRate, sizeof(float)); // Read the learning rate

        // read activation functions from file
        file.read((char *)&actfnSize, sizeof(int)); // Read the size of the activation functions vector
        for (int i = 0; i < actfnSize; ++i)
        {
            file.read((char *)&stringsize, sizeof(int)); // Read the size of the activation function string
            actfn.resize(stringsize); // Resize the string to the correct size
            file.read(&actfn[0], stringsize); // Read the activation function
            this->activationFunctionsStr.push_back(actfn);
        }

        // read shape of the neural network
        int shapeSize;

        file.read((char *)&shapeSize, sizeof(int)); // Read the size of the shape vector
        for (int i = 0; i < shapeSize; ++i)
        {
            int shapeVal;
            file.read((char *)&shapeVal, sizeof(int)); // Read the shape vector
            this->shape.push_back(shapeVal);
        }

        // read weights and biases based on the shape
        for (int i = 0; i < this->shape.size() - 1; ++i)
        {
            MatrixXf weight(this->shape[i + 1], this->shape[i]);
            MatrixXf bias(this->shape[i + 1], 1);
            file.read((char *)weight.data(), weight.size() * sizeof(float)); // Read the weights matrix
            file.read((char *)bias.data(), bias.size() * sizeof(float));     // Read the biases matrix
            this->weights.push_back(weight);
            this->biases.push_back(bias);
        }
        file.close();
    }

    void printWeights()
    {
        for (int i = 0; i < weights.size(); ++i)
        {
            cout << "Weights " << i << ":" << endl;
            printMatrix(weights[i]);
            cout << "---------" << endl;
        }
    }

    void trainOne(const MatrixXf &input, const MatrixXf &expected)
    {
        vector<array<MatrixXf, 2>> outputs = this->detQuery(input);
        MatrixXf output, prev, delta, e_adjust, transposed_weights, derivative;
        output = outputs.back()[0];     // Get the last output
        derivative = outputs.back()[1]; // Get the derivative of the last output
        MatrixXf error = expected - output;
        for (int i = weights.size() - 1; i >= 0; --i)
        {
            transposed_weights = weights[i].transpose(); // Transpose the weights for backpropagation
            // delta = lr * e * output * (1 - output) * input^T
            output = outputs[i + 1][0];     // Get the output of the current layer
            derivative = outputs[i + 1][1]; // Get the derivative of the current layer
            prev = outputs[i][0];           // Get the input of the current layer
            // delta = this->learningRate * (error.cwiseProduct(output).cwiseProduct(oneMat(output.rows()) - output));
            delta = this->learningRate * (error.cwiseProduct(derivative)); // Calculate the delta for the current layer
            // for nullary delta = lr * e.cwise(output).cwise(derivative(unfunction(output)))
            // for relu unfunction doesnt matter
            biases[i] += delta;                 // Update biases
            delta = delta * prev.transpose();   // Matrix multiplication for weight update
            weights[i] += delta;                // Update weights
            error = transposed_weights * error; // Backpropagation error
            // e_adjust = MatrixXf::Ones(transposed_weights.rows(), 1); // Initialize e_adjust to ones
            // for (int j = 0; j < transposed_weights.rows(); ++j) {
            //     e_adjust(j, 0) = transposed_weights.cols() / sumRow(transposed_weights, j); // Sum the weights for the current neuron
            // }
            // error = error.cwiseProduct(e_adjust); // Adjust error for the next layer
        }
    }

    void train(vector<MatrixXf> data, vector<MatrixXf> Expected)
    {
        for (int i = 0; i < data.size(); i++)
        {
            MatrixXf input = data[i];
            MatrixXf expected = Expected[i];
            this->trainOne(input, expected);
        }
    }

    vector<array<MatrixXf, 2>> detQuery(const MatrixXf &input)
    {
        vector<array<MatrixXf, 2>> outputs;
        array<MatrixXf, 2> output = {input, input}; // MatrixXf::Zero(1)};//THIS IS THE ERROR on MATRIXXf::zero
        MatrixXf temp;
        outputs.push_back(output);
        for (int i = 0; i < weights.size(); ++i)
        {
            temp = (weights[i] * output[0]) + biases[i];
            output[0] = (weights[i] * output[0]) + biases[i];             // Matrix multiplication and bias addition
            output[1] = fnInPlace(temp, activationFunctionsDict[this->activationFunctionsStr[i]][0]); // Apply derivative function
            output[0] = fnInPlace(output[0], activationFunctionsDict[this->activationFunctionsStr[i]][1]);
            outputs.push_back(output);
        }
        return outputs;
    }

    MatrixXf query(const MatrixXf &input)
    {
        return this->detQuery(input).back()[0]; // Get the last output
    }

    void trainMNIST(string imagepath = "Original Dataset/train-images.idx3-ubyte", string labelpath = "Original Dataset/train-labels.idx1-ubyte")
    {
        ifstream trainingdata(imagepath, ios::binary);
        ifstream traininglabels(labelpath, ios::binary);
        trainingdata.seekg(16, ios::beg);
        traininglabels.seekg(8, ios::beg);

        unsigned char data[784];
        unsigned char label[1];
        Matrix<float, 784, 1> image;
        Matrix<float, 10, 1> labelmat;

        for (int dataset = 0; dataset < 60000; ++dataset)
        {
            image.setZero();
            labelmat.setZero();
            trainingdata.read((char *)data, 784);
            traininglabels.read((char *)label, 1);
            for (int i = 0; i < 784; ++i)
            {
                image(i, 0) = (((float)((int)data[i]) / 255.0f) * 0.99f) + 0.01f; // Normalize to [0.01, 0.99]
            }
            // this->learningRate = max(0.05f * (dataset / 60000.0f), 0.01f); // Decrease learning rate over time
            labelmat((int)label[0], 0) = 0.99f; // Set the label to 0.99 for the correct class
            this->trainOne(image, labelmat);
            cout << "\r" << "Training Image (" << dataset + 1 << " / 60000)"; // Print progress
        }
        cout << endl
             << "Finished training on 60000 images" << endl;
        trainingdata.close();
        traininglabels.close();
    }

    void testMNIST(string imagepath = "Original Dataset/t10k-images.idx3-ubyte", string labelpath = "Original Dataset/t10k-labels.idx1-ubyte")
    {
        ifstream testingdata(imagepath, ios::binary);
        ifstream testinglabels(labelpath, ios::binary);
        testingdata.seekg(16, ios::beg);
        testinglabels.seekg(8, ios::beg);

        unsigned char data[784];
        unsigned char label[1];
        Matrix<float, 784, 1> image;
        Matrix<float, 10, 1> labelmat;
        int count = 0;

        MatrixXf output;

        for (int dataset = 0; dataset < 10000; ++dataset)
        {
            image.setZero();
            labelmat.setZero();
            testingdata.read((char *)data, 784);
            testinglabels.read((char *)label, 1);
            for (int i = 0; i < 784; ++i)
            {
                image(i, 0) = (((float)((int)data[i]) / 255.0f) * 0.99f) + 0.01f; // Normalize to [0.01, 0.99]
            }
            labelmat((int)label[0], 0) = 0.99f; // Set the label to 0.99 for the correct class
            cout << "\r" << "Testing on image (" << dataset + 1 << " / 10000)";
            output = this->query(image); // Get the output of the neural network
            if (getIndexOfMax(output) == (int)label[0])
            {
                count++;
            }
        }

        testingdata.close();
        testinglabels.close();
        cout << endl
             << "Finished testing on 10000 images" << endl;
        cout << count << " images were classified correctly" << endl;
    }

    void saveNN(string path)
    {
        // Storage format {float learningRate, int stringvectorsize, {int stringsize, string activationfunction...}, int shapeSize, {int shape[i]...} , {float weights...}, {float biases...}}
        // Size of weight and bias matrices implicitly defined by the shape vector
        ofstream file(path, ios::binary);
        if (!file)
        {
            cerr << "Error opening file for writing: " << path << endl;
            return;
        }

        file.write((char *)&this->learningRate, sizeof(float)); // Write the learning rate

        // write activation functions to file
        int actfnSize = this->activationFunctionsStr.size();
        int stringsize;

        file.write((char *)&actfnSize, sizeof(int)); // Write the size of the activation functions vector
        for (const string &actfn : this->activationFunctionsStr)
        {
            stringsize = actfn.size();
            
            file.write((char *)&stringsize, sizeof(int)); // Write the size of the activation function string
            file.write(actfn.c_str(), stringsize); // Write the activation function
        }

        // write shape
        int shapeSize = this->shape.size();
        file.write((char *)&shapeSize, sizeof(int)); // Write the size of the shape vector
        for (int i = 0; i < shapeSize; ++i)
        {
            file.write((char *)&this->shape[i], sizeof(int)); // Write the shape vector
        }

        // write weights and biases to file
        for (int i = 0; i < this->weights.size(); ++i)
        {
            file.write((char *)this->weights[i].data(), this->weights[i].size() * sizeof(float)); // Write the weights matrix
            file.write((char *)this->biases[i].data(), this->biases[i].size() * sizeof(float));   // Write the biases matrix
        }

        file.close();
    }
};

int main(int argc, char *argv[])
{
    NeuralNetwork nn({784, 100, 10, 10}, 0.01f, {"sigmoid"});


    // nn.trainMNIST(); // Train the neural network on MNIST dataset
    // nn.trainMNIST();
    // nn.trainMNIST();

    NeuralNetwork nn2("nn.bin"); // Load the neural network from a file
    nn2.testMNIST();

    // nn.saveNN("nn.bin"); // Save the neural network to a file
    // //Testing
    // nn.testMNIST(); // Test the neural network on MNIST dataset

    // TODO create trial for different learning rates and NN shapes
    // TODO move specific NN functions into the class and move general functions such as mnist out

    cout << "Program Terminated..." << endl;
    return 0;
}