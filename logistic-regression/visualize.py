import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split

plt.style.use("ggplot")

rcParams['figure.figsize'] = 12, 8

data=pd.read_csv("data.csv",index_col="t")
print(data.head())

feature = data[["x1", "x2"]].values
class_label = data.index.values

class_c0 = (class_label == 1).reshape(100, 1)
class_c1 = (class_label == 0).reshape(100, 1)

ax = sns.scatterplot(x = feature[class_c0[:, 0], 0],
                     y = feature[class_c0[:, 0], 1],
                     marker = "^",
                     color = "green",
                     s = 60)
sns.scatterplot(x = feature[class_c1[:, 0], 0],
                y = feature[class_c1[:, 0], 1],
                marker = "X",
                color = "red",
                s = 60)

ax.set(xlabel="x1", ylabel="x2")
ax.legend(["C0", "C1"])
plt.show();

data = pd.read_csv("data.csv")
data = data.reset_index()
X = data[["x1", "x2"]]
y = data["t"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=67
)

X_train_df = X_train.copy()
X_train_df["feature_1"] = X_train_df["x1"].map(lambda x: 1/x)
feature = X_train_df.values
mean_scores = np.mean(feature, axis=0)
std_scores = np.std(feature, axis=0)

theta = np.array([1.43114447, 2.00917464, 3.74590586, -2.38426254])

fig, ax = plt.subplots(figsize=(12, 8))
all_x1 = X["x1"].values
all_x2 = X["x2"].values

mask0 = (y == 0).values
mask1 = (y == 1).values
ax.scatter(all_x1[mask0], all_x2[mask0], marker='X', color='red',   s=60, label='C0 (t=0)')
ax.scatter(all_x1[mask1], all_x2[mask1], marker='^', color='green', s=60, label='C1 (t=1)')

x1_vals = np.linspace(all_x1.min() - 5, all_x1.max() + 5, 500)
x1_vals = x1_vals[x1_vals != 0]


z_x1     = (x1_vals       - mean_scores[0]) / std_scores[0]
z_inv_x1 = (1.0 / x1_vals - mean_scores[2]) / std_scores[2]


z_x2 = -(theta[0] + theta[1] * z_x1 + theta[3] * z_inv_x1) / theta[2]


x2_boundary = z_x2 * std_scores[1] + mean_scores[1]

ax.plot(x1_vals, x2_boundary, color='blue', linewidth=2, label='Decision Boundary')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Scatter Plot with Decision Boundary (basis: 1/x1)")
ax.legend()
ax.set_ylim(all_x2.min() - 10, all_x2.max() + 10)
plt.grid(True)
plt.show()
