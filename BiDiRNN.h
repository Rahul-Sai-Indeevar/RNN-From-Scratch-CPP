#pragma once
#include "RNNLayer.h"
#include "DenseLayer.h"
#include <vector>
#include <algorithm>

class BiDiRNN
{
private:
    RNNLayer forward_rnn;
    RNNLayer backward_rnn;
    DenseLayer output_layer;
    int hidden_sz;

public:
    BiDiRNN(int input_sz, int hidden_size, int output_sz)
        : hidden_sz(hidden_size),
          forward_rnn(input_sz, hidden_size),
          backward_rnn(input_sz, hidden_size),
          output_layer(hidden_size * 2, output_sz) {}

    double train_chunk(const std::vector<Matrix> &x_chunk, const std::vector<Matrix> &y_true_chunk, Matrix &h_forward_prev, double learning_rate){
        int k = x_chunk.size();
        std::vector<Matrix> x_chunk_reversed = x_chunk;
        std::reverse(x_chunk_reversed.begin(), x_chunk_reversed.end());

        auto h_fwd_seq = forward_rnn.forward(x_chunk, h_forward_prev);
        Matrix h_bwd_prev = Matrix::zeros(hidden_sz, 1);
        std::vector<Matrix> h_bwd_seq_reversed = backward_rnn.forward(x_chunk_reversed, h_bwd_prev);
        std::vector<Matrix> h_bwd_seq = h_bwd_seq_reversed;
        std::reverse(h_bwd_seq.begin(), h_bwd_seq.end());
        std::vector<Matrix> h_concat_seq;
        for (int t = 0; t < k; ++t)
        {
            h_concat_seq.push_back(Matrix::vstack(h_fwd_seq[t], h_bwd_seq[t]));
        }
        std::vector<Matrix> y_pred_seq = output_layer.forward(h_concat_seq);
        double chunk_loss = 0.0;
        std::vector<Matrix> dy_sequence;
        for (int t = 0; t < k; ++t)
        {
            Matrix diff = y_pred_seq[t] - y_true_chunk[t];
            chunk_loss += (diff.l2_norm() * diff.l2_norm());
            dy_sequence.push_back(diff * (2.0 / k));
        }
        std::vector<Matrix> dh_concat_seq = output_layer.backward(dy_sequence, learning_rate);

        std::vector<Matrix> dh_fwd_seq(k, Matrix(hidden_sz, 1));
        std::vector<Matrix> dh_bwd_seq(k, Matrix(hidden_sz, 1));

        for (int t = 0; t < k; ++t)
        {
            Matrix dh_f(hidden_sz, 1);
            Matrix dh_b(hidden_sz, 1);
            for (int i = 0; i < hidden_sz; ++i)
                dh_f(i, 0) = dh_concat_seq[t](i, 0);
            for (int i = 0; i < hidden_sz; ++i)
                dh_b(i, 0) = dh_concat_seq[t](i + hidden_sz, 0);
            dh_fwd_seq[t] = dh_f;
            dh_bwd_seq[t] = dh_b;
        }

        std::vector<Matrix> dx_fwd = forward_rnn.backward(dh_fwd_seq, learning_rate);

        std::reverse(dh_bwd_seq.begin(), dh_bwd_seq.end());
        std::vector<Matrix> dx_bwd_reversed = backward_rnn.backward(dh_bwd_seq, learning_rate);
        std::vector<Matrix> dx_bwd = dx_bwd_reversed;
        std::reverse(dx_bwd.begin(), dx_bwd.end());

        return chunk_loss / k;
    }
};