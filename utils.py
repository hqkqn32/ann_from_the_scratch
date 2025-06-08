import numpy as np

import numpy as np

def save_model(model, path):
    weights = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "weights"):
            weights[f"W{i}"] = layer.weights
            weights[f"b{i}"] = layer.biases
    np.savez(path, **weights)

    
def load_model(model, path):
    data = np.load(path)
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "weights"):
            layer.weights = data[f"W{i}"]
            layer.biases = data[f"b{i}"]
    return model