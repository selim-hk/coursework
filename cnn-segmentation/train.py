import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.vgg import vgg16
from torchvision.utils import make_grid

from config import args, device, root
from data import mean, std, test_loader, train_loader
from models import FCN8, FCN32
from utils import Colorize, add_padding, dense_crf, get_color_map, get_parameters, label_accuracy_score


def train_net(net, resume=False):
    criterion = nn.CrossEntropyLoss(ignore_index=21)
    colorize = Colorize(21, get_color_map())
    best_valid_iou = 0

    if resume:
        checkpoint = torch.load(ckpt_path, weights_only = False)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_valid_iou = checkpoint['best_valid_iou']
        print(f'Resume training from epoch {epoch+1}')
    else:
        epoch = 0

    while epoch < args.epoch:
        t1 = time.time()
        saved_images, saved_labels = [], []

        net.train()

        loss_total = 0
        ious = []
        pixel_accs = []

        for batch_idx, (image, label) in enumerate(train_loader):
            if len(saved_images) < 4:
                saved_images.append(image.cpu())
                saved_labels.append(add_padding(label.cpu()))

            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = net(image)
            loss =criterion(output, label)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(output, dim = 1)

            target = label.squeeze(1).cpu().numpy()

            acc, mean_iu = label_accuracy_score(target, pred.cpu().detach().numpy(), n_class=21)

            pixel_accs.append(acc)
            ious.append(mean_iu)

            loss_total += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch : {epoch} || {batch_idx}/{len(train_loader)} || \
                 loss : {loss.item():.3f}, iou : {mean_iu * 100:.3f} pixel_acc : {acc * 100:.3f}')

        total_ious = np.array(ious).T
        total_ious = np.nanmean(total_ious).mean()
        total_pixel_acc = np.array(pixel_accs).mean()

        writer.add_scalar('train_loss', loss_total / len(train_loader), epoch)
        writer.add_scalar('pixel_acc', total_pixel_acc, epoch)
        writer.add_scalar('mean_iou', total_ious, epoch)

        un_norms, preds, outputs = [], [], []
        for image, label in zip(saved_images, saved_labels):
            image_permuted = image.permute(1, 0, 2, 3)
            un_norm = torch.zeros_like(image_permuted)
            for idx, (im, m, s) in enumerate(zip(image_permuted, mean, std)):
                un_norm[idx] = (im * s) + m
            un_norm = un_norm.permute(1, 0, 2, 3)
            un_norms.append(add_padding(un_norm))

            with torch.no_grad():
                output = net(image.to(device))
                pred = torch.argmax(output, dim=1)
                preds.append(add_padding(pred))

        un_norm = make_grid(torch.cat(un_norms), nrow=2)
        label = make_grid(colorize(torch.stack(saved_labels)), nrow=2)
        pred = make_grid(colorize(torch.stack(preds)), nrow=2)

        writer.add_image('img', un_norm, epoch)
        writer.add_image('gt', label, epoch)
        writer.add_image('pred', pred, epoch)

        t = time.time() - t1
        print(f'>> Epoch : {epoch} || AVG loss : {loss_total / len(train_loader):.3f}, \
         iou : {total_ious * 100:.3f} pixel_acc : {total_pixel_acc * 100:.3f} {t:.3f} secs')

        net.eval()
        saved_images, saved_labels = [], []

        valid_loss_total = 0
        valid_ious = []
        valid_pixel_accs = []

        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(test_loader):
                if len(saved_images) < 4:
                    saved_images.append(image.cpu())
                    saved_labels.append(add_padding(label.cpu()))

                image = image.to(device)
                label = label.to(device)

                output = net(image)
                loss = criterion(output, label)
                pred = torch.argmax(output, dim = 1)

                target = label.squeeze(1).cpu().numpy()
                acc, mean_iu = label_accuracy_score(target, pred.cpu().numpy(), n_class=21)

                valid_loss_total += loss.item()

                valid_pixel_accs.append(acc)
                valid_ious.append(mean_iu)

        total_valid_ious = np.array(valid_ious).T
        total_valid_ious = np.nanmean(total_valid_ious).mean()
        total_valid_pixel_acc = np.array(valid_pixel_accs).mean()

        writer.add_scalar('valid_train_loss', valid_loss_total / len(test_loader), epoch)
        writer.add_scalar('valid_pixel_acc', total_valid_pixel_acc, epoch)
        writer.add_scalar('valid_mean_iou', total_valid_ious, epoch)

        un_norms, preds, pred_crfs, outputs = [], [], [], []
        for image, label in zip(saved_images, saved_labels):
            image_permuted = image.permute(1, 0, 2, 3)
            un_norm = torch.zeros_like(image_permuted)
            for idx, (im, m, s) in enumerate(zip(image_permuted, mean, std)):
                un_norm[idx] = (im * s) + m
            un_norm = un_norm.permute(1, 0, 2, 3)
            un_norms.append(add_padding(un_norm))

            with torch.no_grad():
                output = net(image.to(device))
                outputs.append(add_padding(output))
                pred = torch.argmax(output, dim=1)
                preds.append(add_padding(pred))

            output_softmax = torch.nn.functional.softmax(output, dim=1).detach().cpu()
            un_norm_int = (un_norm * 255).squeeze().permute(1, 2, 0).numpy().astype(np.ubyte)
            pred_crf = dense_crf(un_norm_int, output_softmax.squeeze().numpy())
            pred_crfs.append(add_padding(torch.argmax(torch.Tensor(pred_crf), dim=0)).unsqueeze(0))

        valid_un_norm = make_grid(torch.cat(un_norms), nrow=2)
        valid_label = make_grid(colorize(torch.stack(saved_labels)), nrow=2)
        valid_pred = make_grid(colorize(torch.stack(preds)), nrow=2)
        valid_pred_crf = make_grid(colorize(torch.stack(pred_crfs)), nrow=2)

        writer.add_image('valid_img', valid_un_norm, epoch)
        writer.add_image('valid_gt', valid_label, epoch)
        writer.add_image('valid_pred', valid_pred, epoch)
        writer.add_image('valid_pred_crf', valid_pred_crf, epoch)

        print(f'>> Epoch : {epoch} || AVG valid loss : {valid_loss_total / len(test_loader):.3f}, \
          iou : {total_valid_ious * 100:.3f} pixel_acc : {total_valid_pixel_acc * 100:.3f} {t:.3f} secs')

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_valid_iou': best_valid_iou
        }
        torch.save(checkpoint, ckpt_path)

        if total_valid_ious > best_valid_iou:
            best_valid_iou = total_valid_ious
            torch.save(net.state_dict(), ckpt_dir / 'best.pt')

        epoch += 1
    print(f'>> Best validation set iou: {best_valid_iou}')


num_trial=0
result_dir= Path(root) / 'results'
parent_dir = result_dir / f'trial_{num_trial}'
while parent_dir.is_dir():
    num_trial = int(parent_dir.name.replace('trial_',''))
    parent_dir = result_dir / f'trial_{num_trial+1}'

parent_dir = result_dir / 'trial_0'
print(f'Logs and ckpts will be saved in : {parent_dir}')

log_dir = parent_dir
ckpt_dir = parent_dir
ckpt_path = parent_dir / 'model.pt'
writer = SummaryWriter(log_dir)

pretrained_path = Path(root) / 'pretrained_vgg.pt'
pretrained_VGG = vgg16().to(device)
pretrained_VGG.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only = False))

#model = FCN8().to(device)
model = FCN32().to(device)

model.load_pretrained(pretrained_VGG)

optimizer = SGD([{'params': get_parameters(model, True), 'lr': args.lr * 2, 'weight_decay': 0},
                 {'params': get_parameters(model, False)}],
                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

train_net(model, resume=False)
