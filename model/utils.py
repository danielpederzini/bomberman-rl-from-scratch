import cupy as cp

ACTIVATION_FUNCTIONS = {
    "linear": lambda x: x,
    "relu": lambda x: cp.maximum(0, x),
    "sigmoid": lambda x: 1 / (1 + cp.exp(-x)),
    "tanh": lambda x: cp.tanh(x)
}

def initialize_weights(input_size: int, num_neurons: int) -> cp.ndarray:
    """Initialize weights using He initialization"""
    limit = cp.sqrt(2 / input_size)
    return cp.random.normal(0, limit, (input_size, num_neurons))
