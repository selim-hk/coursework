# Coursework

| | Task | Data | Methods |
|---|---|---|---|
| [cnn-classification](cnn-classification) | Image classification | CIFAR-10 | MLP, ConvNet, ResNet (plain / bottleneck), Inception |
| [cnn-segmentation](cnn-segmentation) | Semantic segmentation | SBD (PASCAL VOC 2011) | FCN-32s, FCN-8s (+ DenseCRF) |
| [bias-variance](bias-variance) | Regression, model selection | synthetic | Ridge with polynomial / Fourier / Gaussian bases |
| [logistic-regression](logistic-regression) | Binary classification | 2-D | Logistic regression + basis function, gradient descent |
| [neural-network](neural-network) | Binary classification | 2-D | NumPy MLP with manual backprop |
| [vae](vae) | Generative modeling | 2-D | NumPy VAE with manual backprop and Adam |

```bash
pip install -r requirements.txt
cd cnn-classification && python train.py
```
