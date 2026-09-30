# CIFAR-10 Classification with CNN Blocks

MLP, stacked conv blocks, plain residual blocks, bottleneck residual blocks and Inception blocks, trained on CIFAR-10. Every model uses the same 3-stage backbone.

```
models/
  blocks.py     MLPBlock, ConvBlock, ResBlockPlain, ResBlockBottleneck, InceptionBlock
  networks.py   MyNetworkExample, MyNetwork
config.py       hyper-parameters
data.py         CIFAR-10 loaders
utils.py        weight init
train.py        trains all five models, then evaluates the best checkpoints
```

```bash
python train.py
```

## Results

Test-set results after 100 epochs of SGD (lr 0.1, momentum 0.9, milestones 50/80):

| Model | Blocks | Params | Test acc |
|---|---|---|---|
| MLP | – | 1,649,354 | 62.63% |
| Conv | 10-10-10 | 510,426 | 81.56% |
| ResPlain | 5-5-5 | 510,426 | 89.00% |
| ResBottleneck | 5-5-5 | 113,946 | 86.72% |
| Inception | 5-5-5 | 124,026 | 83.06% |
