import pandas as pd

data = pd.read_csv("data.csv")
X = data[["x1", "x2"]].values
y = data["t"].values
