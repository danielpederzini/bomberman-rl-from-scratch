import cupy as cp
from model.network import DQNetwork

class AdamWOptimizer:
    def __init__(self, network: DQNetwork, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0, max_grad_norm: float = 1.0):
        self.network = network
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self._first_moments: list[cp.ndarray] = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self._second_moments: list[cp.ndarray] = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self._first_moments_b: list[cp.ndarray] = [cp.zeros_like(layer.biases) for layer in self.network.layers]
        self._second_moments_b: list[cp.ndarray] = [cp.zeros_like(layer.biases) for layer in self.network.layers]
        self.step_count: int = 0
    
    def _clip_gradients(self):
        """Clip gradients to prevent exploding gradients."""
        total_norm = 0.0
        for layer in self.network.layers:
            if hasattr(layer, 'weights_grad') and layer.weights_grad is not None:
                grad_norm = float(cp.linalg.norm(layer.weights_grad))
                total_norm += grad_norm ** 2
            if hasattr(layer, 'biases_grad') and layer.biases_grad is not None:
                grad_norm = float(cp.linalg.norm(layer.biases_grad))
                total_norm += grad_norm ** 2

        total_norm = total_norm ** 0.5
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for layer in self.network.layers:
                if hasattr(layer, 'weights_grad') and layer.weights_grad is not None:
                    layer.weights_grad *= clip_coef
                if hasattr(layer, 'biases_grad') and layer.biases_grad is not None:
                    layer.biases_grad *= clip_coef

    def step(self):
        self.step_count += 1
        self._clip_gradients()

        for idx, layer in enumerate(self.network.layers):
            if layer.weights_grad is not None:
                self._first_moments[idx] = self.beta1 * self._first_moments[idx] + (1 - self.beta1) * layer.weights_grad
                self._second_moments[idx] = self.beta2 * self._second_moments[idx] + (1 - self.beta2) * cp.square(layer.weights_grad)

                first_moment_corrected = self._first_moments[idx] / (1 - self.beta1 ** self.step_count)
                second_moment_corrected = self._second_moments[idx] / (1 - self.beta2 ** self.step_count)

                layer.weights -= self.learning_rate * first_moment_corrected / (cp.sqrt(second_moment_corrected) + self.epsilon)
                layer.weights -= self.learning_rate * self.weight_decay * layer.weights

            if layer.biases_grad is not None:
                self._first_moments_b[idx] = self.beta1 * self._first_moments_b[idx] + (1 - self.beta1) * layer.biases_grad
                self._second_moments_b[idx] = self.beta2 * self._second_moments_b[idx] + (1 - self.beta2) * cp.square(layer.biases_grad)

                first_moment_b_corrected = self._first_moments_b[idx] / (1 - self.beta1 ** self.step_count)
                second_moment_b_corrected = self._second_moments_b[idx] / (1 - self.beta2 ** self.step_count)

                layer.biases -= self.learning_rate * first_moment_b_corrected / (cp.sqrt(second_moment_b_corrected) + self.epsilon)

    def reset(self):
        self._first_moments = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self._second_moments = [cp.zeros_like(layer.weights) for layer in self.network.layers]
        self._first_moments_b = [cp.zeros_like(layer.biases) for layer in self.network.layers]
        self._second_moments_b = [cp.zeros_like(layer.biases) for layer in self.network.layers]
        self.step_count = 0