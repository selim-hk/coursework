import numpy as np

from data import X, y
from model import MyModel

lr_cands = [0.001, 0.01, 0.1]
iteration_cands = [1000, 2000, 4000]
width_cands = [4, 8, 16]

for lr in lr_cands:
    for n_iterations in iteration_cands:
        for width in width_cands:
            model_with_relu = MyModel(n_features=2, lr=lr, n_iterations=n_iterations, width=width, Relu=True)
            model_with_tanh = MyModel(n_features=2, lr=lr, n_iterations=n_iterations, width=width, Relu=False)

            model_with_relu.fit(X, y)
            model_with_tanh.fit(X, y)

            accuracy_relu = np.mean(model_with_relu.predict(X) == y)
            accuracy_tanh = np.mean(model_with_tanh.predict(X) == y)
            print (f"lr: {lr}, iterations: {n_iterations}, width: {width}, accuracy with ReLU: {accuracy_relu:.20f}, accuracy with Tanh: {accuracy_tanh:.20f}")

for num in model_with_relu.h2:
    print(num)

lr = 0.1
iterations = 2000
width = 8

for i in range(5):
    perm = np.random.permutation(len(X))
    X_train, X_test = X[perm[:80]], X[perm[80:]]
    y_train, y_test = y[perm[:80]], y[perm[80:]]

    model_relu = MyModel(n_features=2, lr=lr, n_iterations=iterations, width=width, Relu=True)
    model_tanh = MyModel(n_features=2, lr=lr, n_iterations=iterations, width=width, Relu=False)

    model_relu.fit(X_train, y_train)
    model_tanh.fit(X_train, y_train)

    accuracy_relu = np.mean(model_relu.predict(X_test) == y_test)
    accuracy_tanh = np.mean(model_tanh.predict(X_test) == y_test)

    print (f"accuracy with ReLU: {accuracy_relu:.20f}, accuracy with Tanh: {accuracy_tanh:.20f}")

accuracy_relu = np.mean(model_relu.predict(X_train) == y_train)
accuracy_tanh = np.mean(model_tanh.predict(X_train) == y_train)
print (f"accuracy with ReLU on training data: {accuracy_relu:.20f}, accuracy with Tanh on training data: {accuracy_tanh:.20f}")
