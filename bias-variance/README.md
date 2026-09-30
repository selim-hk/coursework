# Bias–Variance and Model Selection

Ridge regression with polynomial, Fourier, Gaussian and identity bases. Hyper-parameters (basis, number of features, λ) are chosen on a validation split and evaluated on a held-out test split. The data is drawn from y = σ(x) + ε with ε ~ N(0, 2²).

```
config.py     search space
features.py   basis functions
data.py       synthetic dataset
train.py      train / validation / test pipeline
demo.py       fitted functions and their mean over resampled datasets, for several λ
```

```bash
python train.py
python demo.py
```
