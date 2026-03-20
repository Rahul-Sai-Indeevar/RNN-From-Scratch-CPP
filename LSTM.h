#pragma once
#include "Matrix.h"
#include <vector>

class LSTM{
private:
    int hidden_size;
    double clip_threshold = 5.0;
    Matrix W_xf, W_hf, b_f; // Forget Gate (f)
    Matrix W_xi, W_hi, b_i; // Input Gate (i)
    Matrix W_xc, W_hc, b_c; // Cell Candidate (c_tilde)
    Matrix W_xo, W_ho, b_o; // Output Gate (o)
    Matrix W_yh, b_y;       // Output Layer Weights
public:
    LSTM(int input_size, int hid_size, int output_size): hidden_size(hid_size), 
        W_xf(Matrix::random(hid_size, input_size)), W_hf(Matrix::random(hid_size, hid_size)), b_f(Matrix::zeros(hid_size, 1)),
        W_xi(Matrix::random(hid_size, input_size)), W_hi(Matrix::random(hid_size, hid_size)), b_i(Matrix::zeros(hid_size, 1)),
        W_xc(Matrix::random(hid_size, input_size)), W_hc(Matrix::random(hid_size, hid_size)), b_c(Matrix::zeros(hid_size, 1)),
        W_xo(Matrix::random(hid_size, input_size)), W_ho(Matrix::random(hid_size, hid_size)), b_o(Matrix::zeros(hid_size, 1)),
        W_yh(Matrix::random(output_size, hid_size)), b_y(Matrix::zeros(output_size, 1)) {
        for (size_t i = 0; i < b_f.data.size(); ++i)
            b_f.data[i] = 1.0;
        }
        int get_hidden_size() const { return hidden_size; }
        double train_chunk(const std::vector<Matrix> x_chunk, const std::vector<Matrix> y_true_chunk, Matrix &h_prev, Matrix &c_prev, double learning_rate){
            int k = x_chunk.size();
            std::vector<Matrix> h_cache, c_cache;
            std::vector<Matrix> f_cache, i_cache, c_tilde_cache, o_cache;
            std::vector<Matrix> y_pred_cache;
            h_cache.push_back(h_prev);
            c_cache.push_back(c_prev);
            double chunk_loss = 0.0;
            for(int t=0;t<k;++t){
                Matrix x_t = x_chunk[t];
                Matrix h_t_1 = h_cache[t];
                Matrix c_t_1 = c_cache[t];
                // Forget gate
                Matrix f_t = Matrix::sigmoid(W_xf.dot(x_t) + W_hf.dot(h_t_1) + b_f);
                f_cache.push_back(f_t);
                // Input gate
                Matrix i_t = Matrix::sigmoid(W_xi.dot(x_t)+ W_hi.dot(h_t_1) + b_i);
                i_cache.push_back(i_t);
                // Cell Candidate (input node)
                Matrix c_tilde_t = Matrix::tanh(W_xc.dot(x_t) + W_hc.dot(h_t_1) + b_c);
                c_tilde_cache.push_back(c_tilde_t);
                // Cell state
                Matrix c_t = f_t.hadamard(c_t_1) + i_t.hadamard(c_tilde_t);
                c_cache.push_back(c_t);
                // Ouput gate
                Matrix o_t = Matrix::sigmoid(W_xo.dot(x_t) + W_ho.dot(h_t_1) + b_o);
                o_cache.push_back(o_t);
                // Next hidden state
                Matrix h_t = o_t.hadamard(Matrix::tanh(c_t));
                h_cache.push_back(h_t);
                // Prediction
                Matrix y_t = W_yh.dot(h_t) + b_y;
                y_pred_cache.push_back(y_t);
                // Loss calc
                Matrix diff = y_t - y_true_chunk[t];
                chunk_loss += (diff.l2_norm() * diff.l2_norm());
            }
            h_prev = h_cache.back();
            c_prev = c_cache.back();
            // Back Prop
            Matrix dW_xf = Matrix::zeros(W_xf.rows, W_xf.cols), dW_hf = Matrix::zeros(W_hf.rows, W_hf.cols), db_f = Matrix::zeros(b_f.rows, 1);
            Matrix dW_xi = Matrix::zeros(W_xi.rows, W_xi.cols), dW_hi = Matrix::zeros(W_hi.rows, W_hi.cols), db_i = Matrix::zeros(b_i.rows, 1);
            Matrix dW_xc = Matrix::zeros(W_xc.rows, W_xc.cols), dW_hc = Matrix::zeros(W_hc.rows, W_hc.cols), db_c = Matrix::zeros(b_c.rows, 1);
            Matrix dW_xo = Matrix::zeros(W_xo.rows, W_xo.cols), dW_ho = Matrix::zeros(W_ho.rows, W_ho.cols), db_o = Matrix::zeros(b_o.rows, 1);
            Matrix dW_yh = Matrix::zeros(W_yh.rows, W_yh.cols), db_y = Matrix::zeros(b_y.rows, 1);

            Matrix dh_next = Matrix::zeros(hidden_size, 1);
            Matrix dc_next = Matrix::zeros(hidden_size, 1);
            for(int t=k-1;t>=0;--t){
                Matrix x_t = x_chunk[t];
                Matrix h_t_1 = h_cache[t]; // prev hidden state
                Matrix c_t_1 = c_cache[t]; // prev cell state
                Matrix c_t = c_cache[t+1]; // current cell state

                Matrix dy_t = (y_pred_cache[t] - y_true_chunk[t]) * (2.0 / k); // error
                dW_yh = dW_yh + dy_t.dot(h_cache[t + 1].transpose());
                db_y = db_y + dy_t;

                // Gradients entering from top (loss) + future step
                Matrix dh_t = W_yh.transpose().dot(dy_t) + dh_next;

                // Backprop through h_t = o_t * tanh(c_t)
                Matrix do_t = dh_t.hadamard(Matrix::tanh(c_t));
                Matrix dc_t = dh_t.hadamard(o_cache[t]).hadamard(Matrix::tanh_derivative(Matrix::tanh(c_t))) + dc_next;

                // Backprop through gates
                Matrix df_t = dc_t.hadamard(c_t_1);
                Matrix di_t = dc_t.hadamard(c_tilde_cache[t]);
                Matrix dc_tilde_t = dc_t.hadamard(i_cache[t]);

                // Gate activations derivatives
                Matrix df_raw = df_t.hadamard(Matrix::sigmoid_derivative(f_cache[t]));
                Matrix di_raw = di_t.hadamard(Matrix::sigmoid_derivative(i_cache[t]));
                Matrix dc_tilde_raw = dc_tilde_t.hadamard(Matrix::tanh_derivative(c_tilde_cache[t]));
                Matrix do_raw = do_t.hadamard(Matrix::sigmoid_derivative(o_cache[t]));

                // Accumulate Weight Gradients
                dW_xf = dW_xf + df_raw.dot(x_t.transpose());
                dW_hf = dW_hf + df_raw.dot(h_t_1.transpose());
                db_f = db_f + df_raw;
                dW_xi = dW_xi + di_raw.dot(x_t.transpose());
                dW_hi = dW_hi + di_raw.dot(h_t_1.transpose());
                db_i = db_i + di_raw;
                dW_xc = dW_xc + dc_tilde_raw.dot(x_t.transpose());
                dW_hc = dW_hc + dc_tilde_raw.dot(h_t_1.transpose());
                db_c = db_c + dc_tilde_raw;
                dW_xo = dW_xo + do_raw.dot(x_t.transpose());
                dW_ho = dW_ho + do_raw.dot(h_t_1.transpose());
                db_o = db_o + do_raw;

                // Gradients to pass to the previous time step
                dh_next = W_hf.transpose().dot(df_raw) + W_hi.transpose().dot(di_raw) + W_hc.transpose().dot(dc_tilde_raw) + W_ho.transpose().dot(do_raw);
                dc_next = dc_t.hadamard(f_cache[t]);
            }
            // Gradient clipping
            auto clip = [&](Matrix &m){
                double norm = m.l2_norm();
                if (norm > clip_threshold) m = m * (clip_threshold / norm);
            };
            clip(dW_xf);
            clip(dW_hf);
            clip(db_f);
            clip(dW_xi);
            clip(dW_hi);
            clip(db_i);
            clip(dW_xc);
            clip(dW_hc);
            clip(db_c);
            clip(dW_xo);
            clip(dW_ho);
            clip(db_o);
            clip(dW_yh);
            clip(db_y);

            // Update weights
            W_xf = W_xf - (dW_xf * learning_rate);
            W_hf = W_hf - (dW_hf * learning_rate);
            b_f = b_f - (db_f * learning_rate);
            W_xi = W_xi - (dW_xi * learning_rate);
            W_hi = W_hi - (dW_hi * learning_rate);
            b_i = b_i - (db_i * learning_rate);
            W_xc = W_xc - (dW_xc * learning_rate);
            W_hc = W_hc - (dW_hc * learning_rate);
            b_c = b_c - (db_c * learning_rate);
            W_xo = W_xo - (dW_xo * learning_rate);
            W_ho = W_ho - (dW_ho * learning_rate);
            b_o = b_o - (db_o * learning_rate);
            W_yh = W_yh - (dW_yh * learning_rate);
            b_y = b_y - (db_y * learning_rate);

            return chunk_loss / k;
        }
};