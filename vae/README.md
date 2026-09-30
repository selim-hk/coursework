# Variational Autoencoder from Scratch

A NumPy VAE on 2-D data:
- **Encoder:** two 8-unit ReLU layers that output the mean and log-variance.
- **Latent sampling:** the reparameterization trick.
- **Decoder:** two 8-unit ReLU layers with a learned per-dimension observation variance.
- **Training:** all gradients are derived by hand. The model minimizes the negative ELBO with Adam.

```
model.py       VAE: encode, decode, loss, manual backprop, generate
optim.py       Adam
visualize.py   original vs. generated samples
train.py       training and sampling
```

```bash
python train.py
```
