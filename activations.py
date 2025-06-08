import numpy as np

def relu(x):

    return np.maximum(0, x)

def relu_derivative(x):
   
    return (x > 0).astype(float)

def sigmoid(x):
   
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    
    return sigmoid_output * (1 - sigmoid_output)

def tanh(x):
   
    return np.tanh(x)

def tanh_derivative(tanh_output):
    
    return 1 - tanh_output * tanh_output

def softmax(x):
    
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softmax_derivative(softmax_output):
    
    return softmax_output * (1 - softmax_output)


