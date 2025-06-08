import numpy as np

data = np.load("gender_model.npz")


for key in data.files:
    print(f"\n🔑 {key}:")
    print(data[key])


import pandas as pd

for key in data.files:
    df = pd.DataFrame(data[key])
    print(f"\n🔑 {key} (shape: {data[key].shape}):")
    print(df)
