# Semantic Segmentation with FCN

FCN-32s and FCN-8s on a VGG16 backbone, trained on SBD (21 PASCAL VOC classes). Optional DenseCRF post-processing refines the predictions.

```
models/
  fcn.py        FCN32, FCN8
config.py       hyper-parameters, device, results dir
data.py         SBD loaders
utils.py        metrics, bilinear init, colorization, DenseCRF, padding
train.py        fine-tunes from pretrained VGG16
evaluate.py     pixel accuracy / mIoU of the best checkpoints
```

```bash
python train.py      # needs pretrained_vgg.pt in this directory
python evaluate.py   # results/trial_0/best.pt (FCN32), results/trial_1/best.pt (FCN8)
```

In `train.py`, choose the model by switching between the `FCN32()` and `FCN8()` lines.

## Results

Validation results after 10 epochs:

| Model | Pixel acc | mIoU |
|---|---|---|
| FCN-32s | 88.24% | 0.599 |
| FCN-8s | 88.51% | 0.612 |
