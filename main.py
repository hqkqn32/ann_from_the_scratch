import utils
import numpy as np  
from activations import ReLU, Sigmoid, Tanh, Softmax
from dense import Dense 
from loses import mse, derrivative_mse, bce, derivative_bce, mae, mae_derivative
from model import Model


X = np.array([
    [160, 50, 22],  # kadın
    [170, 60, 25],  # kadın
    [180, 75, 30],  # erkek
    [175, 85, 28],  # erkek
    [165, 55, 20],  # kadın
    [185, 90, 35],  # erkek
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
model.add(Dense(3, 5))       # 3 giriş → 5 gizli nöron
model.add(ReLU())
model.add(Dense(5, 1))       # 5 → 1 çıkış
model.add(Sigmoid())         # Çünkü çıktı: 0 ya da 1

model.train(
    X, y,
    epochs=1000,
    learning_rate=0.1,
    loss_fn=bce,
    loss_derivative=derivative_bce
)


preds = model.predict(X)

print("\nTahminler:")
for i in range(len(X)):
    gender = "erkek" if preds[i][0] >= 0.5 else "kadın"
    print(f"Girdi: {X[i]}, Tahmin: {preds[i][0]:.3f} → {gender} (Gerçek: {y[i][0]})")
print("\nSon Ağırlıklar:")
