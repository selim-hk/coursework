import torch
from easydict import EasyDict as edict

root = '.'

torch.manual_seed(470)
torch.cuda.manual_seed(470)

args = edict()

args.name = 'main'
args.ckpt_dir = 'ckpts'
args.ckpt_iter = 1000
args.ckpt_reload = 'best'
args.gpu = True

args.num_filters = 16
args.block_type = 'mlp'
args.num_blocks = [5, 5, 5]

args.dataroot = 'dataset/cifar10'
args.batch_size = 128

args.lr = 0.1
args.epoch = 100

args.tensorboard = True
args.log_dir = 'logs'
args.log_iter = 100
