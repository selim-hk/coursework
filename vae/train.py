import numpy as np
import pandas as pd

from model import VAE
from visualize import visualize

D = pd.read_csv("data.csv")
D = D.drop(columns=['t']).values

Z1 = np.random.randn(200, 2) + 5
Z2 = np.random.randn(200, 2) * 2 + 1
X = np.concatenate([Z1, Z2], axis=0)

x_mean = X.mean(0)
x_std = X.std(0)
Xn = (X - x_mean) / x_std
N = Xn.shape[0]

vae = VAE(D, lr=2e-3, epochs=4000)
vae.train(D)

generated_samples = vae.generate(200)
visualize(D, generated_samples)
print(f"  learned obs sigma (standardized) = {np.exp(0.5 * vae.ls).ravel()}")
print(f"  real mean = {D.mean(0)}   std = {D.std(0)}")
print(f"  synth mean= {generated_samples.mean(0)}   std = {generated_samples.std(0)}")
