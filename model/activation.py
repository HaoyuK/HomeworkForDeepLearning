'''
本节实现了sigmoid、relu和tanh三种激活函数
'''
import numpy as np

class ActivationFunction:
    def __init__(self, function_type='sigmoid'):
        self.function_type = function_type
    
    def function(self, x):
        if self.function_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.function_type == 'relu':
            return np.maximum(0, x)
        elif self.function_type == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function type")

    def derivative(self, x):
        if self.function_type == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        elif self.function_type == 'relu':
            return (x > 0).astype(float)
        elif self.function_type == 'tanh':
            return 1 - np.tanh(x)**2
        else:
            raise ValueError("Unsupported activation function type")


if __name__ == "__main__":
    activation = ActivationFunction('sigmoid')
    x = np.array([-1, 0, 1, 2])
    print("Sigmoid values:", activation.function(x))
    print("Sigmoid derivatives:", activation.derivative(x))

    activation.set_function('relu')
    print("ReLU values:", activation.function(x))
    print("ReLU derivatives:", activation.derivative(x))
