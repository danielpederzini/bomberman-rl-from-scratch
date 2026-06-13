import cupy as cp
from model.common import LAYER_TYPES
from model.layer import Layer

LAYER_TYPES["dense"] = Layer

class DQNetwork:
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
    
    def loss(self, predictions: cp.ndarray, targets: cp.ndarray) -> cp.ndarray:
        return cp.mean(cp.square(predictions - targets))

    def backward(self, grad: cp.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
