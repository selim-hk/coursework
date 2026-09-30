import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import args, root
from data import test_dataloader, train_dataloader
from models import MyNetwork, MyNetworkExample
from utils import weight_init

device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)

global_step = 0
best_accuracy = 0.

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
else:
    writer = None


def train_net(net, optimizer, scheduler, block_type, writer):
    global_step = 0
    best_accuracy = 0

    for epoch in range(args.epoch):
        net.train()
        for batch_idx, (x, y) in enumerate(train_dataloader):

            global_step += 1

            x = x.to(device=device)
            y = y.to(device=device)

            logit = net(x)

            accuracy = (logit.argmax(1) == y).float().mean()

            loss = nn.CrossEntropyLoss()(logit, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if global_step % args.log_iter == 0 and writer is not None:
                writer.add_scalar('train_loss', loss, global_step)
                writer.add_scalar('train_accuracy', accuracy, global_step)

            if global_step % args.ckpt_iter == 0:
                torch.save(net.state_dict(), f'{ckpt_dir}/{global_step}.pt')

        net.eval()
        with torch.no_grad():
            test_loss = 0.
            test_accuracy = 0.
            test_num_data = 0.
            for batch_idx, (x, y) in enumerate(test_dataloader):
                x = x.to(device=device)
                y = y.to(device=device)

                logit = net(x)

                loss = nn.CrossEntropyLoss()(logit, y)

                accuracy = (logit.argmax(dim=1) == y).float().mean()

                test_loss += loss.item()*x.shape[0]
                test_accuracy += accuracy.item()*x.shape[0]
                test_num_data += x.shape[0]

            test_loss /= test_num_data
            test_accuracy /= test_num_data

            if writer is not None:
                writer.add_scalar('test_loss', test_loss, global_step)
                writer.add_scalar('test_accuracy', test_accuracy, global_step)

                print(f'Test result of epoch {epoch}/{args.epoch} || loss : {test_loss:.3f} acc : {test_accuracy:.3f} ')

                writer.flush()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), f'{ckpt_dir}/{block_type}_best.pt')

        scheduler.step()
    return best_accuracy


block_types = ['mlp', 'conv','resPlain','resBottleneck','inception']

num_trial=0
parent_dir = result_dir / f'trial_{num_trial}'
while parent_dir.is_dir():
    num_trial = int(parent_dir.name.replace('trial_',''))
    parent_dir = result_dir / f'trial_{num_trial+1}'
print(f'Logs and ckpts will be saved in : {parent_dir}')

networks = []
for block_type in block_types:
    if block_type == 'conv':
        args.num_blocks = [10, 10, 10]
    else:
        args.num_blocks = [5, 5, 5]

    if block_type == 'mlp':
        network = MyNetworkExample(64, block_type).to(device)
    else:
        network = MyNetwork(args.num_filters, block_type, args.num_blocks).to(device)

    network.apply(weight_init)
    networks.append(network)

correct_params = {'mlp' : 1649354, 'conv' : 510426, 'resPlain' : 510426, 'resBottleneck' : 113946, 'inception' : 124026}
for block_type, net  in zip(block_types, networks):
    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'# of parameters in {block_type} net : {num_parameters}')
    print(f'Correct # of parameters in {block_type} net : {correct_params[block_type]}')

final_accs = {}

for block_type, net in zip(block_types, networks):
    try:
        args.name = block_type

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=0.5)

        ckpt_dir = parent_dir / args.name / args.ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir = parent_dir / args.name / args.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        if args.tensorboard:
            writer = SummaryWriter(log_dir)

        t1 = time.time()
        accuracy = train_net(net, optimizer, scheduler, block_type, writer)
        t = time.time()-t1
        print(f'Best test accuracy of {block_type} network : {accuracy:.3f} took {t:.3f} secs')
        final_accs[f'{block_type}'] = accuracy*100
    except Exception as e:
        print(e)

for key in final_accs.keys():
    print(f'Best accuracy of {key} = {final_accs[key]:.2f}%')

block_types = ['mlp', 'conv','resPlain','resBottleneck','inception']
test_accs = {}
test_params= {}

for block_type, net in zip(block_types, networks):
        ckpt_dir = parent_dir / block_type / args.ckpt_dir

        ckpt_path = f'{ckpt_dir}/{block_type}_best.pt'
        try:
            net.load_state_dict(torch.load(ckpt_path))
        except Exception as e:
            print(e)

        net.eval()
        with torch.no_grad():
            test_accuracy = 0.
            test_num_data = 0.
            for batch_idx, (x, y) in enumerate(test_dataloader):
                x = x.to(device=device)
                y = y.to(device=device)

                logit = net(x)

                loss = nn.CrossEntropyLoss()(logit, y)

                accuracy = (logit.argmax(dim=1) == y).float().mean()

                test_accuracy += accuracy.item()*x.shape[0]
                test_num_data += x.shape[0]

            test_accuracy /= test_num_data

            num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)

            test_accs[f'{block_type}'] = test_accuracy*100
            test_params[f'{block_type}'] = num_parameters

correct_accs = {'mlp' : 62.6,'conv' : 81.9,'resPlain' : 88.6, 'resBottleneck' : 86.5, 'inception' : 83.7}
correct_params = {'mlp' : 1649354, 'conv' : 510426, 'resPlain' : 510426, 'resBottleneck' : 113946, 'inception' : 124026}

print(' Method        | Accuracy   | # Params    | Expected Acc | Expected # Params  ')
print('------------------------------------------------------------------------------')
for block in block_types:
        print(f' {block:14}| {str(test_accs[block])[:5]:11}| {str(test_params[block]):11} | {str(correct_accs[block])[:5]:13}| {str(correct_params[block]):12}')
