# Two-Layer Neural Network from Scratch

A NumPy MLP (2 → width → width → 1) with ReLU or tanh hidden layers and a sigmoid output. Backpropagation is written by hand, and training uses full-batch gradient descent.

```
model.py       MyModel: forward, loss, backprop, predict
data.py        dataset
train.py       grid search over lr / iterations / width, then generalization on 80/20 splits
visualize.py   decision boundaries for ReLU and tanh
```

```bash
python train.py
python visualize.py
```
