import numpy as np
from read_data import *

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(pred_classes == true_classes)

if __name__ == "__main__":
    X_train, y_train = load_mnist('./data', kind='train')
    y = y_train[:5]
    predict = [[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.05],
               [0.6,0.3,0.3,0.5,0.6,0.7,0.2,0.124,0.1,0.05],
               [0.2,0.3,0.4,0.5,0.6,0.357,0.8,0.119,0.1,0.05],
               [0.2,0.3,0.4,0.5,0.1236,0.347,0.1238,0.1249,0.1,0.05],
               [0.2,0.3,0.4,0.1235,0.6,0.7,0.8,0.1239,0.1,0.05]]
    print(y)
    print(accuracy(y, predict))

