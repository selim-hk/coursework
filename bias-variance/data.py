import numpy as np

from config import CONFIG
from features import sigmoid

train_xs = 3 * np.random.randn(CONFIG["num_data"])
train_ys = sigmoid(train_xs) + 2 * np.random.randn(CONFIG["num_data"])

indices = np.random.permutation(len(train_xs))
train_xs = train_xs[indices]
train_ys = train_ys[indices]
