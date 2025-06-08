import numpy as np
class Dense:
    def __init__(self,output_size,input_size):
        self.output_size=output_size
        self.input_size=input_size
        self.weights=np.random.randn(input_size, output_size) * 0.01
        self.bias=np.zeros((1, output_size))
        
        
    def forward(self, X):
        self.input=X
        return np.dot(X,self.weights)+self.bias
    
    def backward(self, d_output, learning_rate):
        d_weights = np.dot(self.input.T, d_output)               
        d_bias = np.sum(d_output, axis=0, keepdims=True)         

        self.weights -= learning_rate * d_weights                
        self.biases -= learning_rate * d_bias                    

        d_input = np.dot(d_output, self.weights.T)               
        return d_input                                          

