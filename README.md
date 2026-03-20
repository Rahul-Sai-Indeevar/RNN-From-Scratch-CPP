# Recurrent Neural Networks (RNN) from Scratch in C++

A purely from-scratch implementation of sequence modeling architectures using standard C++ (no external math libraries like Eigen, and no Deep Learning frameworks like PyTorch or TensorFlow). 

This project demonstrates the fundamental mathematics, forward/backward propagation calculus, and state management required to build and train Recurrent Neural Networks.

## 🧠 Architectures Implemented
1. **Vanilla RNN:** The foundational sequence model.
2. **LSTM (Long Short-Term Memory):** Implements Forget, Input, and Output gates with a persistent Cell State to solve the vanishing gradient problem.
3. **GRU (Gated Recurrent Unit):** A streamlined architecture using Update and Reset gates for faster computation.
4. **Deep (Stacked) RNNs:** A modular layer-based approach (`RNNLayer`, `DenseLayer`) allowing for $N$-layer deep sequence networks.
5. **Bi-Directional RNNs:** Processes sequences left-to-right and right-to-left simultaneously, concatenating hidden states to provide full temporal context to the output layer.

## ⚙️ The Core Engine (`Matrix.h`)
To support Backpropagation Through Time (BPTT), a custom linear algebra engine was built from scratch supporting:
* Standard Matrix Multiplication (Dot products)
* Hadamard (Element-wise) Products
* Activation Functions & Derivatives (`Tanh`, `Sigmoid`)
* **L2-Norm Gradient Clipping:** Crucial for preventing the exploding gradient problem inherent in RNNs.

## ⏱️ Truncated BPTT: Regular vs. Randomized
Training RNNs on infinite or extremely long sequences requires **Truncated Backpropagation Through Time (TBPTT)** to prevent memory exhaustion and gradient instability. This project implements and compares two truncation strategies:

### 1. Regular Truncation (Fixed Chunk Size)
* **How it works:** The sequence is split into fixed blocks (e.g., exactly 50 steps). The hidden state is passed forward, but gradients are cut off at the boundary.
* **The Flaw (Boundary Effect):** The network develops a "blind spot" at the boundaries. It never learns the relationship between step 50 and step 51 because the gradient never flows across that line.
* **Why Industry Uses It:** Modern frameworks (PyTorch/TensorFlow) use this because GPUs require static memory allocation. Fixed chunk sizes allow for massive, highly optimized parallel batching.

### 2. Randomized Truncation (Dynamic Chunk Size)
* **How it works:** The chunk size is randomized for every step (e.g., a random length between 20 and 80 steps).
* **The Advantage (Algorithmic Purity):** Because the boundary is constantly shifting every epoch, the network is forced to learn **true temporal invariance**. It eliminates the boundary blind spots and acts as a powerful sequence regularizer.
* **Usage in this Project:** Because this engine runs on the CPU and processes sequences sequentially, we utilize Randomized TBPTT to achieve superior generalization and lower loss without the constraints of GPU batching.

## 🚀 How to Compile and Run
Ensure you have a modern C++ compiler (GCC/Clang) installed.
```bash
    g++ -O3 main.cpp -o rnn_test
    ./rnn_test
```