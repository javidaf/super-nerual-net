import numpy as np


class Activation:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=3):
        return np.where(z > 0, 1, alpha)

    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def linear_derivative(z):
        return np.ones_like(z)

    @staticmethod
    def softmax(z):
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(z):
        # For softmax, this is handled directly in backward pass
        return np.ones_like(z)
