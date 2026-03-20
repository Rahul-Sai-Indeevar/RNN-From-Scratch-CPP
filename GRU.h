#pragma once
#include "Matrix.h"
#include <vector>

class GRU{
private:
    int hidden_size;
    double clip_threshold = 5.0;
    Matrix W_xz, W_hz, b_z; // Update Gate (z)
    Matrix W_xr, W_hr, b_r; // Reset Gate (r)
    Matrix W_xh, W_hh, b_h; // Candidate Hidden State (h_tilde)
    Matrix W_yh, b_y;       // Output Layer Weights
public:
    GRU(int input_size, int hid_size, int output_size) : hidden_size(hid_size),
        W_xz(Matrix::random(hid_size, input_size)), W_hz(Matrix::random(hid_size, hid_size)), b_z(Matrix::zeros(hid_size, 1)),
        W_xr(Matrix::random(hid_size, input_size)), W_hr(Matrix::random(hid_size, hid_size)), b_r(Matrix::zeros(hid_size, 1)),
        W_xh(Matrix::random(hid_size, input_size)), W_hh(Matrix::random(hid_size, hid_size)), b_h(Matrix::zeros(hid_size, 1)),
        W_yh(Matrix::random(output_size, hid_size)), b_y(Matrix::zeros(output_size, 1)) {}
    int get_hidden_size() const { return hidden_size; }
    double train_chunk(const std::vector<Matrix> x_chunk, const std::vector<Matrix> y_true_chunk, Matrix &h_prev, double learning_rate) {
        int k = x_chunk.size();
        std::vector<Matrix> h_cache;
        std::vector<Matrix> z_cache, r_cache, h_tilde_cache, h_reset_cache;
        std::vector<Matrix> y_pred_cache;

        h_cache.push_back(h_prev);
        double chunk_loss = 0.0;
        Matrix ones(hidden_size, 1, 1.0);
        for(int t=0;t<k;++t){
            Matrix x_t = x_chunk[t];
            Matrix h_t_1 = h_cache[t];

            Matrix z_t = Matrix::sigmoid(W_xz.dot(x_t) + W_hz.dot(h_t_1) + b_z);
            z_cache.push_back(z_t);

            Matrix r_t = Matrix::sigmoid(W_xr.dot(x_t) + W_hr.dot(h_t_1) + b_r);
            r_cache.push_back(r_t);

            Matrix h_reset_t = r_t.hadamard(h_t_1);
            h_reset_cache.push_back(h_reset_t);

            Matrix h_tilde_t = Matrix::tanh(W_xh.dot(x_t) + W_hh.dot(h_reset_t) + b_h);
            h_tilde_cache.push_back(h_tilde_t);

            Matrix h_t = z_t.hadamard(h_tilde_t) + (ones - z_t).hadamard(h_t_1);
            h_cache.push_back(h_t);

            Matrix y_t = W_yh.dot(h_t) + b_y; // Prediction
            y_pred_cache.push_back(y_t);
            Matrix diff = y_t - y_true_chunk[t]; // Loss calculation
            chunk_loss += (diff.l2_norm() * diff.l2_norm());
        }
        h_prev = h_cache.back();
        // BackProp
        Matrix dW_xz = Matrix::zeros(W_xz.rows, W_xz.cols), dW_hz = Matrix::zeros(W_hz.rows, W_hz.cols), db_z = Matrix::zeros(b_z.rows, 1);
        Matrix dW_xr = Matrix::zeros(W_xr.rows, W_xr.cols), dW_hr = Matrix::zeros(W_hr.rows, W_hr.cols), db_r = Matrix::zeros(b_r.rows, 1);
        Matrix dW_xh = Matrix::zeros(W_xh.rows, W_xh.cols), dW_hh = Matrix::zeros(W_hh.rows, W_hh.cols), db_h = Matrix::zeros(b_h.rows, 1);
        Matrix dW_yh = Matrix::zeros(W_yh.rows, W_yh.cols), db_y = Matrix::zeros(b_y.rows, 1);
        Matrix dh_next = Matrix::zeros(hidden_size,1);
        for(int t = k-1;t>=0;--t){
            Matrix x_t = x_chunk[t];
            Matrix h_t_1 = h_cache[t];

            Matrix dy_t = (y_pred_cache[t] - y_true_chunk[t]) * (2.0/k);
            dW_yh = dW_yh + dy_t.dot(h_cache[t+1].transpose());
            db_y = db_y + dy_t;

            Matrix dh_t = W_yh.transpose().dot(dy_t) + dh_next;
            Matrix dz_t = dh_t.hadamard(h_tilde_cache[t] - h_t_1);
            Matrix dh_tilde_t = dh_t.hadamard(z_cache[t]);
            Matrix dh_t_1_direct = dh_t.hadamard(ones - z_cache[t]);
            Matrix dh_tilde_raw = dh_tilde_t.hadamard(Matrix::tanh_derivative(h_tilde_cache[t]));

            dW_xh = dW_xh + dh_tilde_raw.dot(x_t.transpose());
            dW_hh = dW_hh + dh_tilde_raw.dot(h_reset_cache[t].transpose());
            db_h = db_h + dh_tilde_raw;

            Matrix dh_reset = W_hh.transpose().dot(dh_tilde_raw);
            Matrix dr_t = dh_reset.hadamard(h_t_1);
            Matrix dh_t_1_r = dh_reset.hadamard(r_cache[t]);
            
            Matrix dr_raw = dr_t.hadamard(Matrix::sigmoid_derivative(r_cache[t]));
            Matrix dz_raw = dz_t.hadamard(Matrix::sigmoid_derivative(z_cache[t]));

            dW_xz = dW_xz + dz_raw.dot(x_t.transpose());
            dW_hz = dW_hz + dz_raw.dot(h_t_1.transpose());
            db_z = db_z + dz_raw;

            dW_xr = dW_xr + dr_raw.dot(x_t.transpose());
            dW_hr = dW_hr + dr_raw.dot(h_t_1.transpose());
            db_r = db_r + dr_raw;

            dh_next = dh_t_1_direct + dh_t_1_r + W_hz.transpose().dot(dz_raw) + W_hr.transpose().dot(dr_raw);
        }
        // Gradient clipping
        auto clip = [&](Matrix &m){
            double norm = m.l2_norm();
            if (norm > clip_threshold) m = m * (clip_threshold / norm);
        };
        clip(dW_xz);
        clip(dW_hz);
        clip(db_z);
        clip(dW_xr);
        clip(dW_hr);
        clip(db_r);
        clip(dW_xh);
        clip(dW_hh);
        clip(db_h);
        clip(dW_yh);
        clip(db_y);
        // updation
        W_xz = W_xz - (dW_xz * learning_rate);
        W_hz = W_hz - (dW_hz * learning_rate);
        b_z = b_z - (db_z * learning_rate);
        W_xr = W_xr - (dW_xr * learning_rate);
        W_hr = W_hr - (dW_hr * learning_rate);
        b_r = b_r - (db_r * learning_rate);
        W_xh = W_xh - (dW_xh * learning_rate);
        W_hh = W_hh - (dW_hh * learning_rate);
        b_h = b_h - (db_h * learning_rate);
        W_yh = W_yh - (dW_yh * learning_rate);
        b_y = b_y - (db_y * learning_rate);

        return chunk_loss / k;
    }
};