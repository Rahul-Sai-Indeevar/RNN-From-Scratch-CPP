#pragma once
#include "Matrix.h"
#include <vector>

class RNNLayer{
private:
    Matrix W_hx, W_hh, b_h;
    double clip_threshold = 5.0;
    std::vector<Matrix> x_cache;
    std::vector<Matrix> h_cache;
public:
    int input_size;
    int hidden_size;
    RNNLayer(int input_sz, int hidden_sz) : input_size(input_sz), hidden_size(hidden_sz), 
        W_hx(Matrix::random(hidden_sz, input_sz)),
        W_hh(Matrix::random(hidden_sz, hidden_sz)),
        b_h(Matrix::zeros(hidden_sz, 1)) {}

    std::vector<Matrix> forward(const std::vector<Matrix> &x_chunk, Matrix &h_prev){
        x_cache = x_chunk;
        h_cache.clear();
        h_cache.push_back(h_prev);
        std::vector<Matrix> h_out_sequence;
        for(int t=0;t<x_chunk.size();++t){
            Matrix h_t = Matrix::tanh(W_hx.dot(x_chunk[t]) +  W_hh.dot(h_cache[t]) + b_h);
            h_cache.push_back(h_t);
            h_out_sequence.push_back(h_t);
        }
        h_prev = h_cache.back();
        return h_out_sequence;
    }

    std::vector<Matrix> backward(const std::vector<Matrix>& dh_sequence, double learning_rate){
        int k = x_cache.size();
        std::vector<Matrix> dx_sequence(k,Matrix::zeros(input_size,1));
        Matrix dW_hx = Matrix::zeros(W_hx.rows, W_hx.cols);
        Matrix dW_hh = Matrix::zeros(W_hh.rows, W_hh.cols);
        Matrix db_h = Matrix::zeros(b_h.rows, 1);

        Matrix dh_next = Matrix::zeros(hidden_size, 1);
        for(int t=k-1;t>=0;--t){
            Matrix dh_t = dh_sequence[t] + dh_next;
            Matrix dh_raw = dh_t.hadamard(Matrix::tanh_derivative(h_cache[t+1]));
            dW_hx = dW_hx + dh_raw.dot(x_cache[t].transpose());
            dW_hh = dW_hh + dh_raw.dot(h_cache[t].transpose());
            db_h = db_h + dh_raw;
            dx_sequence[t] = W_hx.transpose().dot(dh_raw);
            dh_next = W_hh.transpose().dot(dh_raw);
        }
        // Clipping gradients
        auto clip = [&](Matrix &m){
            double norm = m.l2_norm();
            if (norm > clip_threshold) m = m * (clip_threshold / norm);
        };
        clip(dW_hx);
        clip(dW_hh);
        clip(db_h);

        // Updating weights
        W_hx = W_hx - (dW_hx * learning_rate);
        W_hh = W_hh - (dW_hh * learning_rate);
        b_h = b_h - (db_h * learning_rate);

        return dx_sequence;
    }
};