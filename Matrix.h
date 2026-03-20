#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <cassert>
#include <stdexcept>

class Matrix{
public:
    int rows, cols;
    std::vector<double> data;
    Matrix() : rows(0), cols(0), data() {}
    Matrix(int r, int c, double init_val = 0.0) : rows(r), cols(c), data(r * c, init_val) {}

    inline double &operator()(int r, int c) { return data[r * cols + c]; }
    inline const double &operator()(int r, int c) const { return data[r * cols + c]; }

    Matrix operator+(const Matrix &other) const{
        assert(rows == other.rows && cols == other.cols);
        Matrix res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            res.data[i] = data[i] + other.data[i];
        return res;
    }

    Matrix operator-(const Matrix &other) const{
        assert(rows == other.rows && cols == other.cols);
        Matrix res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            res.data[i] = data[i] - other.data[i];
        return res;
    }

    Matrix operator*(double scalar) const{
        Matrix res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            res.data[i] = data[i] * scalar;
        return res;
    }

    Matrix dot(const Matrix &other) const{
        assert(cols == other.rows);
        Matrix res(rows, other.cols);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < other.cols; ++j){
                double sum = 0.0;
                for (int k = 0; k < cols; ++k){
                    sum += (*this)(i, k) * other(k, j);
                }
                res(i, j) = sum;
            }
        }
        return res;
    }

    Matrix transpose() const{
        Matrix res(cols, rows);
        for (int i = 0; i < rows; ++i){
            for (int j = 0; j < cols; ++j){
                res(j, i) = (*this)(i, j);
            }
        }
        return res;
    }

    // Hadamard (Element-wise) Product
    Matrix hadamard(const Matrix &other) const{
        assert(rows == other.rows && cols == other.cols);
        Matrix res(rows, cols);
        for (size_t i = 0; i < data.size(); ++i)
            res.data[i] = data[i] * other.data[i];
        return res;
    }

    // L2 Norm for Gradient Clipping
    double l2_norm() const{
        double sum = 0.0;
        for (double val : data)
            sum += val * val;
        return std::sqrt(sum);
    }

    // Activations
    static Matrix tanh(const Matrix &m){
        Matrix res(m.rows, m.cols);
        for (size_t i = 0; i < m.data.size(); ++i)
            res.data[i] = std::tanh(m.data[i]);
        return res;
    }
    static Matrix tanh_derivative(const Matrix &m){
        Matrix res(m.rows, m.cols);
        for (size_t i = 0; i < m.data.size(); ++i)
            res.data[i] = 1.0 - (m.data[i] * m.data[i]);
        return res;
    }

    static Matrix sigmoid(const Matrix &m){
        Matrix res(m.rows, m.cols);
        for (size_t i = 0; i < m.data.size(); ++i)
            res.data[i] = 1.0 / (1.0 + std::exp(-m.data[i]));
        return res;
    }
    static Matrix sigmoid_derivative(const Matrix &m){
        Matrix res(m.rows, m.cols);
        for (size_t i = 0; i < m.data.size(); ++i)
            res.data[i] = m.data[i] * (1.0 - m.data[i]);
        return res;
    }

    static Matrix zeros(int r, int c) { return Matrix(r, c, 0.0); }

    static Matrix random(int r, int c, double min_val = -0.1, double max_val = 0.1){
        Matrix res(r, c);
        std::random_device rd;
        std::mt19937 gen(1337);
        std::uniform_real_distribution<> dis(min_val, max_val);
        for (size_t i = 0; i < res.data.size(); ++i)
            res.data[i] = dis(gen);
        return res;
    }

    // Concatenates two column vectors vertically
    static Matrix vstack(const Matrix &top, const Matrix &bottom){
        assert(top.cols == 1 && bottom.cols == 1);
        Matrix res(top.rows + bottom.rows, 1);
        for (int i = 0; i < top.rows; ++i)
            res(i, 0) = top(i, 0);
        for (int i = 0; i < bottom.rows; ++i)
            res(i + top.rows, 0) = bottom(i, 0);
        return res;
    }
};