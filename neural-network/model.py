import numpy as np


def logistic_function(x):
    return 1/ (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)


class MyModel:
    def __init__(self, n_features, lr = 0.01, n_iterations = 100, width = 4, Relu = True):
        self.W1 = np.ones((width, n_features))
        self.b1 = np.zeros((width, 1))
        self.z1 = None
        self.h1 = None
        self.W2 = np.ones((width, width))
        self.b2 = np.zeros((width, 1))
        self.z2 = None
        self.h2 = None
        self.w  = np.random.randn(width, 1)
        self.b0 = np.random.rand(1, 1)
        self.zout = None
        self.lr = lr
        self.epochs = n_iterations
        self.X = None
        self.Relu = Relu
        self.mean = None
        self.std = None


    def forward(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.mean = mean
        self.std = std
        X = (X - mean) / std
        self.X = X
        self.z1 = self.W1 @ X.T + self.b1
        if self.Relu:
            self.h1 = Relu(self.z1)
        else:
            self.h1 = tanh(self.z1)
        self.z2 = self.W2 @ self.h1 + self.b2
        if self.Relu:
            self.h2 = Relu(self.z2)
        else:
            self.h2 = tanh(self.z2)
        self.zout = self.w.T @ self.h2 + self.b0
        return logistic_function(self.zout)

    def loss(self, y_pred, y):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss  = (-1)*(np.dot(y, np.log(y_pred.T)) + np.dot((1-y), np.log(1 - y_pred.T))) / y.shape[0]
        return loss


    def fit(self, X, y):
        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            L = self.loss(y_pred, y)

            dy_pred = ((1 - y.T) / ( 1 - y_pred)) - (y.T / y_pred)
            dzout = dy_pred * logistic_function(self.zout) * (1 - logistic_function(self.zout))
            dw = dzout * self.h2
            db0 = dzout
            dh2 = dzout * self.w
            if self.Relu:
                dz2 = dh2 * (self.z2 > 0)
            else:
                dz2 = dh2 * (1 - self.h2**2)
            dh1 = self.W2.T @ dz2
            dW2 = dz2 @ self.h1.T / X.shape[0]
            db2 = dz2
            if self.Relu:
                dz1 = dh1 * (self.z1 > 0)
            else:
                dz1 = dh1 * (1 - self.h1**2)
            dW1 = dz1 @ self.X / X.shape[0]
            db1 = dz1

            dw = dw.mean(axis=1, keepdims=True)
            db0 = db0.mean(axis=1, keepdims=True)
            db2 = db2.mean(axis=1, keepdims=True)
            db1 = db1.mean(axis=1, keepdims=True)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.w -= self.lr * dw
            self.b0 -= self.lr * db0


    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        y_pred = self.forward(X)
        return y_pred.flatten()
