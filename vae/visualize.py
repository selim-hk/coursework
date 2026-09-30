import matplotlib.pyplot as plt


def visualize(X_original, X_generated):
    plt.figure(figsize=(7, 5))
    plt.scatter(X_original[:, 0],  X_original[:, 1],  c='green', alpha=0.6, label='Original')
    plt.scatter(X_generated[:, 0], X_generated[:, 1], c='red',   alpha=0.6, label='Generated')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Original vs. generated samples')
    plt.legend()
    plt.show()
