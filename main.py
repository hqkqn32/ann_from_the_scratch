import numpy as np  
from activations import ReLU, Sigmoid
from dense import Dense 
from loses import bce, derivative_bce
from model import Model
import utils  

X = np.array([
    [160, 50, 22],  # woman
    [170, 60, 25],  # woman
    [180, 75, 30],  # man
    [175, 85, 28],  # man
    [165, 55, 20],  # woman
    [185, 90, 35],  # man
])

y = np.array([
    [0],
    [0],
    [1],
    [1],
    [0],
    [1]
])

X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

model = Model()
model.add(Dense(3, 5))     
model.add(ReLU())         
model.add(Dense(5, 1))    
model.add(Sigmoid())        

model.train(
    X, y,
    epochs=1000,
    learning_rate=0.1,
    loss_fn=bce,
    loss_derivative=derivative_bce
)

preds = model.predict(X)

for i in range(len(X)):
    gender = "man" if preds[i][0] >= 0.5 else "woman"




utils.save_model(model, "gender_model.npz")

