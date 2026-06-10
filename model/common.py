from typing import Dict

LAYER_TYPES: Dict[str, type] = {}

def register_layer(layer_type: str, layer_class: type):
    """Register a layer type for use in Network"""
    LAYER_TYPES[layer_type] = layer_class

def initialize_weights(input_size: int, num_neurons: int):
    """Import from utils to avoid circular dependency"""
    from utils import initialize_weights as _init_weights
    return _init_weights(input_size, num_neurons)