# Coursework

| | Task | Dataset | Models |
|---|---|---|---|
| [cnn-classification](cnn-classification) | Image classification | CIFAR-10 | MLP, ConvNet, ResNet (plain / bottleneck), Inception |
| [cnn-segmentation](cnn-segmentation) | Semantic segmentation | SBD (PASCAL VOC 2011) | FCN-32s, FCN-8s (+ DenseCRF) |

```bash
pip install -r requirements.txt
cd cnn-classification && python train.py
```
