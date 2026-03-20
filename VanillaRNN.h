#pragma once
#include "Matrix.h"
#include <vector>

class VanillaRNN{
private:
    Matrix W_hx, W_hh, W_yh;
    Matrix b_h, b_y;
    int hidden_size;
    double clip_threshold = 5.0;
public:
    VanillaRNN(int input_size, int hid_size, int output_size)
        : hidden_size(hid_size),
          W_hx(Matrix::random(hid_size, input_size)),
          W_hh(Matrix::random(hid_size, hid_size)),
          W_yh(Matrix::random(output_size, hid_size)),
          b_h(Matrix::zeros(hid_size, 1)),
          b_y(Matrix::zeros(output_size, 1)) {}

    int get_hidden_size() const { return hidden_size; }
    double train_chunk(const std::vector<Matrix> x_chunk, const std::vector<Matrix> y_true_chunk, Matrix& h_prev, double learning_rate){
        int k = x_chunk.size();
        std::vector<Matrix> h_cache;
        std::vector<Matrix> y_pred_cache;
        h_cache.push_back(h_prev);
        double chunk_loss = 0.0;
        for(int t=0;t<k;++t){
            Matrix h_t = Matrix::tanh(W_hx.dot(x_chunk[t]) + W_hh.dot(h_cache[t]) + b_h);
            h_cache.push_back(h_t);
            Matrix y_t = W_yh.dot(h_t) + b_y;
            y_pred_cache.push_back(y_t);
            Matrix diff = y_t - y_true_chunk[t];
            chunk_loss += (diff.l2_norm() * diff.l2_norm());
        }
        h_prev = h_cache.back();
        // BackProp
        Matrix dW_hx = Matrix::zeros(W_hx.rows, W_hx.cols);
        Matrix dW_hh = Matrix::zeros(W_hh.rows, W_hh.cols);
        Matrix dW_yh = Matrix::zeros(W_yh.rows, W_yh.cols);
        Matrix db_h = Matrix::zeros(b_h.rows, b_h.cols);
        Matrix db_y = Matrix::zeros(b_y.rows, b_y.cols);
        Matrix dh_next = Matrix::zeros(hidden_size, 1);
        for(int t=k-1;t>=0;--t){
            Matrix dy_t = (y_pred_cache[t] - y_true_chunk[t]) * (2.0 / k);
            dW_yh = dW_yh + dy_t.dot(h_cache[t + 1].transpose());
            db_y = db_y + dy_t;
            Matrix dh_t = W_yh.transpose().dot(dy_t) + dh_next;
            Matrix dtanh = Matrix::tanh_derivative(h_cache[t + 1]);
            Matrix dh_raw = dh_t.hadamard(dtanh);
            dW_hx = dW_hx + dh_raw.dot(x_chunk[t].transpose());
            dW_hh = dW_hh + dh_raw.dot(h_cache[t].transpose());
            db_h = db_h + dh_raw;
            dh_next = W_hh.transpose().dot(dh_raw);
        }
        clip_gradient(dW_hx);
        clip_gradient(dW_hh);
        clip_gradient(dW_yh);
        clip_gradient(db_h);
        clip_gradient(db_y);
        W_hx = W_hx - (dW_hx * learning_rate);
        W_hh = W_hh - (dW_hh * learning_rate);
        W_yh = W_yh - (dW_yh * learning_rate);
        b_h = b_h - (db_h * learning_rate);
        b_y = b_y - (db_y * learning_rate);

        return chunk_loss / k;
    }
private:
    void clip_gradient(Matrix &grad){
        double norm = grad.l2_norm();
        if (norm > clip_threshold){
            grad = grad * (clip_threshold / norm);
        }
    }
};