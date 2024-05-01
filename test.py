'''
测试部分需支持导入训练好的模型，输出在测试集上的分类准确率（Accuracy）。
'''
import os
import numpy as np
import json
import argparse
from model.linear_model import *
from metric import *
from read_data import *

def load_model_parameters(file_path):
    if os.path.exists(file_path):
        config = file_path.split("/")[-2]
        # 解析参数配置
        layer_sizes = [size for size in config.split('_')]
        # 加载对应参数的模型
        file_name = "{}_{}_{}_{}_{}".format(layer_sizes[0], layer_sizes[1], layer_sizes[2], layer_sizes[3], layer_sizes[4])
        # file_path = os.path.join(directory, file_name)
        activation = ActivationFunction(layer_sizes[-1])  # 这里假设激活函数为ReLU，你可以根据实际情况进行修改
        model = MultiLayerNeuralNetwork(int(layer_sizes[0]), [int(layer_sizes[1]), int(layer_sizes[2])], 10, activation.function, activation.derivative)
        # 加载模型参数
        for i, weights in enumerate(model.weights):
            model.weights[i] = np.load(os.path.join(file_path, f'weights_{i}.npy'))
        for i, biases in enumerate(model.biases):
            model.biases[i] = np.load(os.path.join(file_path, f'biases_{i}.npy'))
        return model
    else:
        raise FileNotFoundError("Configuration file does not exist.")

# 在测试集上进行测试
def test(model, X_test, y_test):
    # 进行测试
    output = model.forward(X_test)
    test_accuracy = accuracy(y_test, output)
    print(f"Test Accuracy: {test_accuracy:.4f}")

def main(args):
    model_dir = args.model_dir
    # 加载最佳模型
    model = load_model_parameters(model_dir)
    # 加载测试集数据
    X_test, y_test = load_mnist('./data', kind='t10k')
    X_test = X_test.reshape(-1, 784)
    # 在测试集上进行测试
    test(model, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using a Trained Model on Test Dataset")
    parser.add_argument('--model_dir', type=str, help='Input feature dimension (M)')
    args = parser.parse_args()
    main(args)


