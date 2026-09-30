import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identical_features(x):
    return np.array(x).reshape(-1, 1)

def polynomial_features(x, num_features):
    return np.array([x**n for n in range(num_features)]).T

def fourier_features(x, num_features):
    return np.array([np.sin(2*np.pi*n*x / num_features) for n in range(num_features)]).T

def gaussian_features(x, num_features):
    return np.array([np.exp(-(x - (2*np.pi*m/num_features - np.pi))**2) for m in range(num_features)]).T
