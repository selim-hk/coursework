from pathlib import Path

import torch
from easydict import EasyDict as edict

root = '.'

torch.backends.cudnn.benchmark = True

torch.manual_seed(470)
torch.cuda.manual_seed(470)

args = edict()
args.batch_size = 1
args.lr = 1e-4
args.momentum = 0.9
args.weight_decay = 5e-4
args.epoch = 10
args.tensorboard = True
args.gpu = True

device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)
