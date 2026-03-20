#include <iostream>
#include <vector>
#include <cmath>
#include "Matrix.h"
#include "VanillaRNN.h"
#include "LSTM.h"
#include "GRU.h"
#include "DeepRNN.h"
#include "BiDiRNN.h"

// Helper to generate a Sine Wave Dataset
void generate_dataset(int length, std::vector<Matrix> &X, std::vector<Matrix> &Y){
    for (int i = 0; i < length; ++i){
        Matrix x(1, 1);
        Matrix y(1, 1);

        // Input: sin(t)
        x(0, 0) = std::sin(0.1 * i);
        // Target: sin(t + 1) -> predicting the next step
        y(0, 0) = std::sin(0.1 * (i + 1));

        X.push_back(x);
        Y.push_back(y);
    }
}

// Sequence Training Loop
void train_sequence(const std::string &method_name, VanillaRNN &rnn,const std::vector<Matrix> &X_data,const std::vector<Matrix> &Y_data,bool use_random_truncation){
    std::cout << "\n--- Starting Training: " << method_name << " ---\n";
    Matrix h_state = Matrix::zeros(rnn.get_hidden_size(), 1);

    int total_length = X_data.size();
    int t = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(20, 80); // Randomized TBPTT chunk size
    int fixed_k = 50;                                // Regular TBPTT chunk size
    double learning_rate = 0.05;

    int next_print = 2000;
    double epoch_loss = 0.0;
    int chunks_processed = 0;

    while (t < total_length){
        int k = use_random_truncation ? dist(rng) : fixed_k;
        if (t + k > total_length)
            k = total_length - t;

        // Slicing chunks
        std::vector<Matrix> x_chunk(X_data.begin() + t, X_data.begin() + t + k);
        std::vector<Matrix> y_chunk(Y_data.begin() + t, Y_data.begin() + t + k);

        // Training chunk
        double loss = rnn.train_chunk(x_chunk, y_chunk, h_state, learning_rate);
        epoch_loss += loss;
        chunks_processed++;

        t += k;

        // Printing for every 2000 steps
        if (t >= next_print || t == total_length){
            std::cout << "Step: " << t << " / " << total_length << " | Avg Chunk Loss: " << (epoch_loss / chunks_processed) << "\n";
            epoch_loss = 0.0;
            chunks_processed = 0;
            next_print += 2000;
        }
    }
}

// Note the bug fix in the print logic: `t >= next_print` instead of `t % 2000 == 0`
void train_lstm_sequence(const std::string &method_name,LSTM &lstm,const std::vector<Matrix> &X_data,const std::vector<Matrix> &Y_data,bool use_random_truncation){
    std::cout << "\n--- Starting Training: " << method_name << " ---\n";

    // LSTM : Both hidden state and cell state tracking
    Matrix h_state = Matrix::zeros(lstm.get_hidden_size(), 1);
    Matrix c_state = Matrix::zeros(lstm.get_hidden_size(), 1);

    int total_length = X_data.size();
    int t = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(20, 80);
    int fixed_k = 50;
    double learning_rate = 0.05;

    double epoch_loss = 0.0;
    int chunks_processed = 0;
    int next_print = 2000;

    while (t < total_length){
        int k = use_random_truncation ? dist(rng) : fixed_k;
        if (t + k > total_length) k = total_length - t;

        std::vector<Matrix> x_chunk(X_data.begin() + t, X_data.begin() + t + k);
        std::vector<Matrix> y_chunk(Y_data.begin() + t, Y_data.begin() + t + k);

        // Pass both h_state and c_state
        double loss = lstm.train_chunk(x_chunk, y_chunk, h_state, c_state, learning_rate);

        epoch_loss += loss;
        chunks_processed++;
        t += k;

        if (t >= next_print || t == total_length){
            std::cout << "Step: " << t << " / " << total_length << " | Avg Chunk Loss: " << (epoch_loss / chunks_processed) << "\n";
            epoch_loss = 0.0;
            chunks_processed = 0;
            next_print += 2000;
        }
    }
}

void train_gru_sequence(const std::string &method_name,GRU &gru,const std::vector<Matrix> &X_data,const std::vector<Matrix> &Y_data,bool use_random_truncation){
    std::cout << "\n--- Starting Training: " << method_name << " ---\n";
    Matrix h_state = Matrix::zeros(gru.get_hidden_size(), 1);

    int total_length = X_data.size();
    int t = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(20, 80);
    int fixed_k = 50;
    double learning_rate = 0.05;

    double epoch_loss = 0.0;
    int chunks_processed = 0;
    int next_print = 2000;

    while (t < total_length){
        int k = use_random_truncation ? dist(rng) : fixed_k;
        if (t + k > total_length) k = total_length - t;

        std::vector<Matrix> x_chunk(X_data.begin() + t, X_data.begin() + t + k);
        std::vector<Matrix> y_chunk(Y_data.begin() + t, Y_data.begin() + t + k);

        double loss = gru.train_chunk(x_chunk, y_chunk, h_state, learning_rate);
        epoch_loss += loss;
        chunks_processed++;

        t += k;

        if (t >= next_print || t == total_length){
            std::cout << "Step: " << t << " / " << total_length << " | Avg Chunk Loss: " << (epoch_loss / chunks_processed) << "\n";
            epoch_loss = 0.0;
            chunks_processed = 0;
            next_print += 2000;
        }
    }
}

void train_deep_sequence(DeepRNN &network,const std::vector<Matrix> &X_data,const std::vector<Matrix> &Y_data){
    std::cout << "\n--- Starting Training: Deep Stacked RNN (2 Layers) ---\n";

    // Hidden states for the deep network
    // Assuming 2 layers of size 16 for simplicity
    std::vector<Matrix> h_states;
    h_states.push_back(Matrix::zeros(16, 1));
    h_states.push_back(Matrix::zeros(16, 1));

    int total_length = X_data.size();
    int t = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(20, 80); // Randomized TBPTT
    double learning_rate = 0.05;

    double epoch_loss = 0.0;
    int chunks_processed = 0;
    int next_print = 2000;

    while (t < total_length){
        int k = dist(rng);
        if (t + k > total_length) k = total_length - t;

        std::vector<Matrix> x_chunk(X_data.begin() + t, X_data.begin() + t + k);
        std::vector<Matrix> y_chunk(Y_data.begin() + t, Y_data.begin() + t + k);

        // Training the Deep Network
        double loss = network.train_chunk(x_chunk, y_chunk, h_states, learning_rate);

        epoch_loss += loss;
        chunks_processed++;
        t += k;

        if (t >= next_print || t == total_length){
            std::cout << "Step: " << t << " / " << total_length << " | Avg Chunk Loss: " << (epoch_loss / chunks_processed) << "\n";
            epoch_loss = 0.0;
            chunks_processed = 0;
            next_print += 2000;
        }
    }
}

void train_birnn_sequence(const std::string &method_name,BiDiRNN &network,const std::vector<Matrix> &X_data,const std::vector<Matrix> &Y_data,bool use_random_truncation){
    std::cout << "\n--- Starting Training: " << method_name << " ---\n";

    // only persist the FORWARD hidden state across chunks
    // The backward state is reset to 0 at the end of each chunk
    Matrix h_forward_prev = Matrix::zeros(16, 1);

    int total_length = X_data.size();
    int t = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(20, 80); // Randomized TBPTT chunk size
    int fixed_k = 50;                                // Regular TBPTT chunk size
    double learning_rate = 0.05;

    double epoch_loss = 0.0;
    int chunks_processed = 0;
    int next_print = 2000;

    while (t < total_length){
        int k = use_random_truncation ? dist(rng) : fixed_k;
        if (t + k > total_length) k = total_length - t;

        // Slicing chunks
        std::vector<Matrix> x_chunk(X_data.begin() + t, X_data.begin() + t + k);
        std::vector<Matrix> y_chunk(Y_data.begin() + t, Y_data.begin() + t + k);

        // Training chunk
        double loss = network.train_chunk(x_chunk, y_chunk, h_forward_prev, learning_rate);

        epoch_loss += loss;
        chunks_processed++;
        t += k;

        // Printing loss
        if (t >= next_print || t == total_length){
            std::cout << "Step: " << t << " / " << total_length << " | Avg Chunk Loss: " << (epoch_loss / chunks_processed) << "\n";
            epoch_loss = 0.0;
            chunks_processed = 0;
            next_print += 2000;
        }
    }
}

int main(){
    int seq_length = 10000;
    std::vector<Matrix> X_data, Y_data;
    generate_dataset(seq_length, X_data, Y_data);

    // Network Architecture: 1 Input -> 16 Hidden -> 1 Output
    VanillaRNN rnn_regular(1, 16, 1);
    VanillaRNN rnn_randomized(1, 16, 1);

    // Train and compare both!
    // Regular Truncation (Fixed size 50)
    train_sequence("Regular Truncated BPTT (Fixed k=50)", rnn_regular, X_data, Y_data, false);

    // Randomized Truncation (Dynamic size 20-80)
    train_sequence("Randomized Truncated BPTT (Dynamic k=20 to 80)", rnn_randomized, X_data, Y_data, true);

    LSTM my_lstm(1, 16, 1);
    train_lstm_sequence("LSTM Randomized TBPTT", my_lstm, X_data, Y_data, true);

    GRU my_gru(1, 16, 1);
    train_gru_sequence("GRU Randomized TBPTT", my_gru, X_data, Y_data, true);

    DeepRNN deep_rnn(1, {16, 16}, 1);
    train_deep_sequence(deep_rnn, X_data, Y_data);

    BiDiRNN my_birnn(1, 16, 1);
    train_birnn_sequence("Bi-Directional RNN (Randomized TBPTT)", my_birnn, X_data, Y_data, true);

    // Comment or un-comment according the models we want to run/test 

    return 0;
}