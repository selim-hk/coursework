import numpy as np
import torch
import torch.nn as nn

from config import device, result_dir
from data import mean, std, test_loader
from models import FCN8, FCN32
from utils import Colorize, dense_crf, get_color_map, label_accuracy_score

FCN32_path = result_dir / 'trial_0' / 'best.pt'
FCN8_path = result_dir / 'trial_1' / 'best.pt'

use_crf = False

model1 = FCN32().to(device)
model1.load_state_dict(torch.load(FCN32_path, map_location=device))
model2 = FCN8().to(device)
model2.load_state_dict(torch.load(FCN8_path, map_location=device))

criterion = nn.CrossEntropyLoss(ignore_index=21)
colorize = Colorize(21, get_color_map())

for net in [model1, model2]:
    net.eval()

    valid_loss_total = 0
    valid_ious = []
    valid_pixel_accs = []
    valid_ious_crf = []
    valid_pixel_accs_crf = []

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)

            output = net(image)
            loss = criterion(output, label)
            pred = torch.argmax(output, dim = 1)

            if use_crf:
                image_permuted = image.cpu().permute(1, 0, 2, 3)
                un_norm = torch.zeros_like(image_permuted)
                for idx, (im, m, s) in enumerate(zip(image_permuted, mean, std)):
                    un_norm[idx] = (im * s) + m
                un_norm = un_norm.permute(1, 0, 2, 3)

                output_softmax = torch.nn.functional.softmax(output, dim=1).detach().cpu()
                un_norm_int = (un_norm * 255).squeeze().permute(1, 2, 0).numpy().astype(np.ubyte)
                pred_crf = dense_crf(un_norm_int, output_softmax.squeeze().numpy())
                pred_crf = np.expand_dims(np.argmax(pred_crf, 0), 0)

                target = label.squeeze(1).cpu().numpy()
                acc_crf, mean_iu_crf = label_accuracy_score(target, pred_crf, n_class=21)
                valid_pixel_accs_crf.append(acc_crf)
                valid_ious_crf.append(mean_iu_crf)

            target = label.squeeze(1).cpu().numpy()
            acc, mean_iu = label_accuracy_score(target, pred.cpu().numpy(), n_class=21)

            valid_loss_total += loss.item()

            valid_pixel_accs.append(acc)
            valid_ious.append(mean_iu)

        total_valid_ious = np.array(valid_ious).T
        total_valid_ious = np.nanmean(total_valid_ious).mean()
        total_valid_pixel_acc = np.array(valid_pixel_accs).mean()

        print(f'{type(net).__name__}:')
        print(f'Pixel accuracy: {total_valid_pixel_acc * 100:.3f}, mIoU: {total_valid_ious:.3f}')

        if use_crf:
            total_valid_ious_crf = np.array(valid_ious_crf).T
            total_valid_ious_crf = np.nanmean(total_valid_ious_crf).mean()
            total_valid_pixel_acc_crf = np.array(valid_pixel_accs_crf).mean()
            print(f'CRF Pixel accuracy: {total_valid_pixel_acc_crf * 100:.3f}, CRF mIoU: {total_valid_ious_crf:.3f}')
