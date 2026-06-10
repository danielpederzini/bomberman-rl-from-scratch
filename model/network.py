import cupy as cp
from common import LAYER_TYPES
from layer import Layer

LAYER_TYPES["dense"] = Layer

class Network:
    def __init__(self, layer_definitions: list[dict]):
        self.layers = self.initialize_layers(layer_definitions)

    def initialize_layers(self, layer_definitions: list[dict]):
        layers = []
        for definition in layer_definitions:
            layer_type = definition.get('type', 'dense')
            layer_class = LAYER_TYPES.get(layer_type)
            if layer_class is None:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            layer = layer_class.from_definition(definition)
            layers.append(layer)
        return layers
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
