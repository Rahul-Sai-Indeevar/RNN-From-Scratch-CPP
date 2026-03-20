#pragma once
#include "RNNLayer.h"
#include "DenseLayer.h"
#include <vector>

class DeepRNN{
private:
    std::vector<RNNLayer> layers;
    DenseLayer output_layer;
public:
    DeepRNN(int input_sz, const std::vector<int> &hidden_sizes, int output_sz) : output_layer(hidden_sizes.back(), output_sz){
        int current_input_sz = input_sz;
        for(int h_sz : hidden_sizes){
            layers.emplace_back(current_input_sz, h_sz);
            current_input_sz = h_sz;
        }
    }

    double train_chunk(const std::vector<Matrix> &x_chunk, const std::vector<Matrix> &y_true_chunk, std::vector<Matrix> &h_states, double learning_rate){
        int k = x_chunk.size();
        std::vector<Matrix> current_sequence = x_chunk;

        for (size_t i = 0; i < layers.size(); ++i){
            current_sequence = layers[i].forward(current_sequence, h_states[i]);
        }
        std::vector<Matrix> y_pred_seq = output_layer.forward(current_sequence);

        double chunk_loss = 0.0;
        std::vector<Matrix> dy_sequence;
        for (int t = 0; t < k; ++t){
            Matrix diff = y_pred_seq[t] - y_true_chunk[t];
            chunk_loss += (diff.l2_norm() * diff.l2_norm());
            dy_sequence.push_back(diff * (2.0 / k));
        }

        std::vector<Matrix> current_gradients = output_layer.backward(dy_sequence, learning_rate);
        for (int i = layers.size() - 1; i >= 0; --i){
            current_gradients = layers[i].backward(current_gradients, learning_rate);
        }

        return chunk_loss / k;
    }
};