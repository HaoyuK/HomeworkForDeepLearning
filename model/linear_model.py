'''
模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度；
'''
import numpy as np 
from model.activation import *

class MultiLayerNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_function, activation_derivative):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = activation_function
        self.activation_derivative = activation_derivative
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases for each layer
        for i in range(len(self.layer_sizes) - 1):
            # weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(1 / self.layer_sizes[i])
            weight = np.random.normal(0, pow(self.layer_sizes[i], -0.5), (self.layer_sizes[i], self.layer_sizes[i + 1]))
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        activation = X
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.activation(z)
            self.activations.append(activation)
        
        return self.activations[-1]

    def backward(self, output, delta, lambda_reg = 0.001):
        m = output.shape[0]
        delta *= self.activation_derivative(self.activations[-1])

        deltas = [delta]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(self.activations[i])
            deltas.insert(0, delta)

        gradients = [(np.dot(self.activations[i].T, deltas[i])  + lambda_reg * self.weights[i],
                      np.sum(deltas[i], axis=0, keepdims=True)  ) for i in range(len(self.weights))] 
        return gradients


if __name__ == "__main__":
    # 测试
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 3], [1, 1, 4]])
    y = np.array([[0], [1], [1], [0]])

    # 定义网络结构
    input_size = 3
    hidden_size = 4
    output_size = 1

    # Example of usage:
    activation = ActivationFunction('sigmoid')
    network = MultiLayerNeuralNetwork(input_size, [hidden_size, hidden_size], output_size, activation.function, activation.derivative)

    print(network.forward(X))

