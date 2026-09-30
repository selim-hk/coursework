from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from config import args

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

train_dataset = CIFAR10(args.dataroot, download=True, train=True, transform=train_transform)
test_dataset = CIFAR10(args.dataroot, download=True, train=False, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
