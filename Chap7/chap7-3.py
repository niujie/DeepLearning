import numpy as np


# 7.3.1 derivative of Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):  # derivative of sigmoid
    return sigmoid(x) * (1 - sigmoid(x))


# 7.3.2 derivative of ReLU
def derivative_ReLU(x):  # derivative of ReLU
    d = np.array(x, copy=True)  # tensor for saving gradients
    d[x < 0] = 0  # derivative is 0 for negative elements
    d[x > 0] = 1  # derivative is 1 for positive elements
    return d


# 7.3.3 derivative of LeakyReLU
def derivative_LeakyReLU(x, p):  # derivative of LeakyReLU
    dx = np.ones_like(x)  # gradients tensor
    dx[x < 0] = p  # derivative is p for negative elements
    return dx


# 7.3.4 derivative of Tanh
def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def derivative_Tanh(x):  # derivative of Tanh
    return 1 - tanh(x) ** 2
