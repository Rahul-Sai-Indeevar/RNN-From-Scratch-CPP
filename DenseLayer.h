#pragma once
#include "Matrix.h"
#include <vector>

class DenseLayer{
private:
    Matrix W_y, b_y;
    std::vector<Matrix> input_cache;
public:
    DenseLayer(int input_size, int output_size) : W_y(Matrix::random(output_size, input_size)), b_y(Matrix::zeros(output_size, 1)) {}

    std::vector<Matrix> forward(const std::vector<Matrix>& h_sequence){
        input_cache = h_sequence;
        std::vector<Matrix> y_sequence;
        for(const auto& h_t : h_sequence){
            y_sequence.push_back(W_y.dot(h_t) + b_y);
        }
        return y_sequence;
    }

    std::vector<Matrix> backward(const std::vector<Matrix>& dy_sequence, double learning_rate){
        int k = input_cache.size();
        std::vector<Matrix> dh_sequence;

        Matrix dW_y = Matrix::zeros(W_y.rows, W_y.cols);
        Matrix db_y = Matrix::zeros(b_y.rows, 1);
        for(int t=0;t<k;++t){
            dh_sequence.push_back(W_y.transpose().dot(dy_sequence[t]));
            dW_y = dW_y + dy_sequence[t].dot(input_cache[t].transpose());
            db_y = db_y + dy_sequence[t];
        }
        W_y = W_y - (dW_y * learning_rate);
        b_y = b_y - (db_y * learning_rate);

        return dh_sequence;
    }
};