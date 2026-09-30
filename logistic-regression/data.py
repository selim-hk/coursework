import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")
data = data.reset_index()
X = data[["x1", "x2"]]
y = data["t"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=67
)

y_train_vals = y_train.values
y_test_vals = y_test.values
