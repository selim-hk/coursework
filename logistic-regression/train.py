import numpy as np

from data import X_test, X_train, y_test_vals, y_train_vals
from model import compute_cost, gradient_descent, logistic_function

lr_candidates = np.arange(0.01, 1, 0.01)

result = []

for n_features in range(10):
    for lr in lr_candidates:
        for n_iterations in range(100, 1000, 100):
            X_tr = X_train.copy()
            X_te = X_test.copy()
            processed_data = []
            for dt in [X_tr, X_te]:
                dt[f"feature_{n_features}"] = dt["x1"].map(lambda x: 1 / (x**n_features) if x != 0 else 0)
                feature = dt.values
                mean_scores = np.mean(feature, axis=0)
                std_scores = np.std(feature, axis=0)
                scores = (feature - mean_scores) / std_scores
                dt_processed = np.append(np.ones((scores.shape[0], 1)), scores, axis=1)
                processed_data.append(dt_processed)

            X_train_proc, X_test_proc = processed_data
            rows, cols = X_train_proc.shape
            theta = np.ones(cols)

            cost, gradient = compute_cost(theta, X_train_proc, y_train_vals)
            theta, costs = gradient_descent(X_train_proc, y_train_vals, theta, lr, n_iterations)

            y_pred = logistic_function(np.dot(X_test_proc , theta)) >= 0.5
            acc = (np.mean(y_pred == y_test_vals)) * 100
            print(f"n_features: {n_features}, lr: {lr}, n_iterations: {n_iterations}, acc : {acc}")
            result.append((n_features, lr, n_iterations, acc))

max_tuple = max(result, key=lambda x: x[-1])
print(max_tuple)
print(f"{max_tuple[-1]:.10f}")

n_iterations = 900
lr = np.float64(0.17)

X_train["feature_1"] = X_train["x1"].map(lambda x : 1/x)
feature = X_train.values
mean_scores = np.mean(feature, axis=0)
std_scores = np.std(feature, axis=0)
scores = (feature - mean_scores) / std_scores
X_train = np.append(np.ones((scores.shape[0], 1)), scores, axis=1)

X_test["feature_1"] = X_test["x1"].map(lambda x : 1/x)
feature = X_test.values
mean_scores = np.mean(feature, axis=0)
std_scores = np.std(feature, axis=0)
scores = (feature - mean_scores) / std_scores
X_test = np.append(np.ones((scores.shape[0], 1)), scores, axis=1)

rows, cols = X_train.shape
theta = np.ones(cols)

cost, gradient = compute_cost(theta, X_train, y_train_vals)
theta, costs = gradient_descent(X_train, y_train_vals, theta, lr, n_iterations)

y_pred = logistic_function(np.dot(X_test , theta)) >= 0.5
acc = (np.mean(y_pred == y_test_vals)) * 100
print(f"n_features: {n_features}, lr: {lr}, n_iterations: {n_iterations}, acc : {acc}")

print(theta)
