import cupy as cp
from model.network import DQNetwork

class AdamWOptimizer:
    def __init__(self, network: DQNetwork, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.network = network
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._first_moments: list[cp.ndarray] = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self._second_moments: list[cp.ndarray] = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self.step_count: int = 0
    
    def step(self):
        self.step_count += 1
        for idx, layer in enumerate(self.network.layers):
            if layer.weights_grad is not None:
                self._first_moments[idx] = self.beta1 * self._first_moments[idx] + (1 - self.beta1) * layer.weights_grad
                self._second_moments[idx] = self.beta2 * self._second_moments[idx] + (1 - self.beta2) * cp.square(layer.weights_grad)
                
                first_moment_corrected = self._first_moments[idx] / (1 - self.beta1 ** self.step_count)
                second_moment_corrected = self._second_moments[idx] / (1 - self.beta2 ** self.step_count)
                
                layer.weights -= self.learning_rate * first_moment_corrected / (cp.sqrt(second_moment_corrected) + self.epsilon)
                layer.weights -= self.learning_rate * self.weight_decay * layer.weights
            
            if layer.biases_grad is not None:
                layer.biases -= self.learning_rate * layer.biases_grad

    def reset(self):
        self._first_moments = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self._second_moments = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self.step_count = 0