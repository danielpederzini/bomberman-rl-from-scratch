import cupy as cp
from typing import Optional, Dict, Any
from model.utils import ACTIVATION_FUNCTIONS, initialize_weights

class Layer:
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray, activation: callable, activation_name: Optional[str] = None):
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.activation_name = activation_name or getattr(activation, '__name__', '')
        self.weights_grad: Optional[cp.ndarray] = None
        self.biases_grad: Optional[cp.ndarray] = None
        self._last_input: Optional[cp.ndarray] = None
        self._last_output: Optional[cp.ndarray] = None

    @staticmethod
    def from_definition(definition: Dict[str, Any]):
        input_size = definition['input_size']
        num_neurons = definition['num_neurons']
        weights = initialize_weights(input_size, num_neurons)
        biases = cp.zeros(num_neurons)
        activation_name = definition.get('activation', 'linear')
        activation = ACTIVATION_FUNCTIONS.get(activation_name, lambda x: x)
        return Layer(weights, biases, activation, activation_name)

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self._last_input = input
        linear_output = cp.dot(input, self.weights) + self.biases
        self._last_output = self.activation(linear_output)
        return self._last_output

    def backward(self, output_grad: cp.ndarray) -> cp.ndarray:
        if self._last_output is not None:
            activation_grad = output_grad
            if self.activation_name == "relu":
                activation_grad = output_grad * (self._last_output > 0)
            elif self.activation_name == "sigmoid":
                activation_grad = output_grad * self._last_output * (1 - self._last_output)
            elif self.activation_name == "tanh":
                activation_grad = output_grad * (1 - self._last_output ** 2)
            output_grad = activation_grad

        self.weights_grad = cp.dot(self._last_input.T, output_grad)
        self.biases_grad = cp.sum(output_grad, axis=0)
        input_grad = cp.dot(output_grad, self.weights.T)
        return input_grad

    def get_last_output(self) -> cp.ndarray | None:
        """Return the last forward pass output for visualization."""
        return self._last_output
