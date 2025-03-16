#include <iostream>
#include <vector>

class Matrix {
    public:
        Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows, std::vector<int>(cols, 0)) {}

        int& operator()(int row, int col) {
            return data_[row][col];
        }

        const int& operator()(int row, int col) const {
            return data_[row][col];
        }

        int rows() const { return rows_; }
        int cols() const { return cols_; }

    private:
        int rows_, cols_;
        std::vector<std::vector<int>> data_;
};


int main() {
    Matrix mat(3, 3);
    mat(0, 0) = 1;
    mat(1, 1) = 2;
    mat(2, 2) = 3;

    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}