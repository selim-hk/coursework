from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import SBDataset
from torchvision.datasets.vision import StandardTransform

from config import args
from utils import toLongTensor

mean = [.485, .456, .406]
std = [.229, .224, .225]

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_train_target = transforms.Compose([
    toLongTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_test_target = transforms.Compose([
    toLongTensor()
])

sbd_transform_train = StandardTransform(transform_train, transform_train_target)
sbd_transform_test = StandardTransform(transform_test, transform_test_target)

try:
    train_dataset = SBDataset(root='.', image_set='train', mode='segmentation', download=True, transforms=sbd_transform_train)
except:
    train_dataset = SBDataset(root='.', image_set='train', mode='segmentation', download=False, transforms=sbd_transform_train)

try:
    test_dataset = SBDataset(root='.', image_set='val', mode='segmentation', download=True, transforms=sbd_transform_test)
except:
    test_dataset = SBDataset(root='.', image_set='val', mode='segmentation', download=False, transforms=sbd_transform_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
