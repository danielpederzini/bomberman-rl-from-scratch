import cupy as cp
from typing import Optional, Dict, Any
from utils import ACTIVATION_FUNCTIONS, initialize_weights

class Layer:
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray, activation: callable):
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.weights_grad: Optional[cp.ndarray] = None
        self.biases_grad: Optional[cp.ndarray] = None
        self._last_input: Optional[cp.ndarray] = None

    @staticmethod
    def from_definition(definition: Dict[str, Any]):
        input_size = definition['input_size']
        num_neurons = definition['num_neurons']
        weights = initialize_weights(input_size, num_neurons)
        biases = cp.zeros(num_neurons)
        activation = ACTIVATION_FUNCTIONS.get(definition.get('activation'), lambda x: x)
        return Layer(weights, biases, activation)

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self._last_input = input
        dot_product = cp.dot(input, self.weights)
        linear_output = dot_product + self.biases
        return self.activation(linear_output)
