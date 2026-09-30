import matplotlib.pyplot as plt
import numpy as np

from data import X, y
from model import MyModel, Relu, logistic_function, tanh

model_relu_full = MyModel(n_features=2, lr=0.1, n_iterations=4000, width=8, Relu=True)
model_tanh_full = MyModel(n_features=2, lr=0.1, n_iterations=4000, width=8, Relu=False)
model_relu_full.fit(X, y)
model_tanh_full.fit(X, y)

print(f'ReLU accuracy on full data: {np.mean(model_relu_full.predict(X) == y):.4f}')
print(f'Tanh accuracy on full data: {np.mean(model_tanh_full.predict(X) == y):.4f}')


margin = 3
x1_min, x1_max = X[:, 0].min() - margin, X[:, 0].max() + margin
x2_min, x2_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                        np.linspace(x2_min, x2_max, 300))
grid = np.c_[xx1.ravel(), xx2.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, model, title in zip(axes, [model_relu_full, model_tanh_full], ['ReLU', 'Tanh']):
    grid_norm = (grid - model.mean) / model.std

    model.z1 = model.W1 @ grid_norm.T + model.b1
    if title == 'ReLU':
        model.h1 = Relu(model.z1)
    else:
        model.h1 = tanh(model.z1)
    model.z2 = model.W2 @ model.h1 + model.b2
    if title == 'ReLU':
        model.h2 = Relu(model.z2)
    else:
        model.h2 = tanh(model.z2)
    model.zout = model.w.T @ model.h2 + model.b0
    probs = logistic_function(model.zout).reshape(xx1.shape)


    cf = ax.contourf(xx1, xx2, probs, levels=np.linspace(0, 1, 50), cmap="viridis", alpha=0.9)
    ax.contour(xx1, xx2, probs, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')

    scatter_colors = ['#E8700A' if yi == 0 else '#1A5BB5' for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=scatter_colors, edgecolors='black', linewidths=0.8, s=45, zorder=5)
    ax.set_title(f'Decision Boundary ({title})', fontsize=15, fontweight='bold')
    ax.set_xlabel('$', fontsize=13)
    ax.set_ylabel('$', fontsize=13)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.set_label('P(y=1)', fontsize=12)

plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()
