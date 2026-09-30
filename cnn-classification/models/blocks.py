import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        output = self.act(self.bn1(self.fc1(x)))
        output = self.act(self.bn2(self.fc2(output)))
        output = self.act(self.bn3(self.fc3(output)))
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1):
        super(ConvBlock, self).__init__()
        self.cn = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.bn(self.cn(x)))
        return output


class ResBlockPlain(nn.Module):
    def __init__(self, in_channels):
        super(ResBlockPlain, self).__init__()
        self.cn1 = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = in_channels)
        self.cn2 = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.bn2(self.cn2(self.relu(self.bn1(self.cn1(x)))))

        output = self.relu(output + x)
        return output


class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(ResBlockBottleneck, self).__init__()
        self.cn1 = nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = hidden_channels)
        self.cn2 = nn.Conv2d(in_channels = hidden_channels, out_channels = hidden_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = hidden_channels)
        self.cn3 = nn.Conv2d(in_channels = hidden_channels, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(num_features = in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        relu1 = self.relu(self.bn1(self.cn1(x)))
        relu2 = self.relu(self.bn2(self.cn2(relu1)))
        output = self.relu(x + self.bn3(self.cn3(relu2)))
        return output


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        assert out_channels%8==0, 'out channel should be mutiplier of 8'

        self.cnn1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels//4, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = out_channels//4)

        self.cnn2 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels//2, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = out_channels//2)
        self.cnn3 = nn.Conv2d(in_channels = out_channels//2, out_channels = out_channels//2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(num_features = out_channels//2)

        self.cnn4 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels//8, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn4 = nn.BatchNorm2d(num_features = out_channels//8)
        self.cnn5 = nn.Conv2d(in_channels = out_channels//8, out_channels = out_channels//8, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.bn5 = nn.BatchNorm2d(num_features = out_channels//8)

        self.mxpl = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        self.cnn6 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels//8, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn6 = nn.BatchNorm2d(num_features = out_channels//8)

        self.relu = nn.ReLU()

    def forward(self, x):
        branch1x1 = self.relu(self.bn1(self.cnn1(x)))
        branch3x3 = self.relu(self.bn3(self.cnn3(self.relu(self.bn2(self.cnn2(x))))))
        branch5x5 = self.relu(self.bn5(self.cnn5(self.relu(self.bn4(self.cnn4(x))))))
        branchmxpool = self.relu(self.bn6(self.cnn6(self.mxpl(x))))

        output = torch.cat([branch1x1, branch3x3, branch5x5, branchmxpool], 1)
        return output
