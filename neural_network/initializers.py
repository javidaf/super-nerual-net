import numpy as np

class Initializer:
    @staticmethod
    def xavier(input_dim, output_dim):
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, (input_dim, output_dim))
    
    @staticmethod
    def he(input_dim, output_dim):
        return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
    
    @staticmethod
    def normal(input_dim, output_dim, scale=0.001):
        return np.random.randn(input_dim, output_dim) * scale
    
