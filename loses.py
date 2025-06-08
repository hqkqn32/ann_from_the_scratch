import numpy as np
def mse(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

def derrivative_mse(y_true,y_pred):
    return 2* (y_pred-y_true)/y_true.size


def bce(y_true, y_pred):
    epsilon = 1e-12  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def derivative_bce(y_true, y_pred):
    epsilon = 1e-12  
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.size

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_derivative(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.size


