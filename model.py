import numpy as np 


class Model:
    def __init__(self):
        self.layers = []

    def add(self,layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def predict(self, X):
        return self.forward(X)

    def backward(self,loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, y, epochs, learning_rate, loss_fn, loss_derivative):
        for epoch in range(epochs):
            output = self.forward(X)                           
            loss = loss_fn(y, output)                             
            loss_grad = loss_derivative(y, output)               
            self.backward(loss_grad, learning_rate)              

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")


