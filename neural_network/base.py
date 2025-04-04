from abc import ABC, abstractmethod

class BaseNeuralNetwork(ABC):
    @abstractmethod
    def forward(self, X):
        pass
    
    @abstractmethod
    def backward(self, gradient):
        pass
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass