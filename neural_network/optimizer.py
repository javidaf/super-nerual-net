import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def initialize(self, weights, biases):
        pass

    def update(self, weights, biases, gradients_w, gradients_b):
        raise NotImplementedError


class SGD(Optimizer):
    def update(self, weights, biases, gradients_w, gradients_b):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * gradients_w[i]
            biases[i] -= self.learning_rate * gradients_b[i]
        return weights, biases


class Adam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
    def initialize(self, weights, biases):
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, gradients_w, gradients_b):
        self.t += 1
        
        for i in range(len(weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]
            
            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i]**2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i]**2)
            
            # Compute bias-corrected moments
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)
            
            weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
        return weights, biases


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        
    def initialize(self, weights, biases):
        self.v_w = [np.zeros_like(w) for w in weights]
        self.v_b = [np.zeros_like(b) for b in biases]
        
    def update(self, weights, biases, gradients_w, gradients_b):
        for i in range(len(weights)):
            # Update accumulated squared gradients
            self.v_w[i] = self.rho * self.v_w[i] + (1 - self.rho) * (gradients_w[i]**2)
            self.v_b[i] = self.rho * self.v_b[i] + (1 - self.rho) * (gradients_b[i]**2)
            
            weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.v_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)
            
        return weights, biases


class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        
    def initialize(self, weights, biases):
        self.G_w = [np.zeros_like(w) for w in weights]
        self.G_b = [np.zeros_like(b) for b in biases]
        
    def update(self, weights, biases, gradients_w, gradients_b):
        for i in range(len(weights)):
            # Accumulate squared gradients
            self.G_w[i] += gradients_w[i]**2
            self.G_b[i] += gradients_b[i]**2
            
            weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.G_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.G_b[i]) + self.epsilon)
            
        return weights, biases