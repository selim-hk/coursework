# Logistic Regression with a Basis Function

Logistic regression trained with batch gradient descent. The added feature 1/x₁ⁿ makes the data linearly separable in feature space. The number of basis terms, the learning rate and the iteration count are grid-searched on a stratified 80/20 split.

```
model.py       sigmoid, cross-entropy cost, gradient descent
data.py        train / test split
train.py       grid search, then the final model
visualize.py   data scatter and decision boundary
```

```bash
python train.py
python visualize.py
```

With the basis 1/x₁, lr 0.17 and 900 iterations, the model reaches 100% accuracy on the test split.
