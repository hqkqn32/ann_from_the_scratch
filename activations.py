import numpy as np

class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out, learning_rate):
        return d_out * (self.input > 0).astype(float)


class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, d_out, learning_rate):
        return d_out * self.output * (1 - self.output)


class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, d_out, learning_rate):
        return d_out * (1 - self.output ** 2)


class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output

    def backward(self, d_out, learning_rate):
        return d_out * (self.output * (1 - self.output))
