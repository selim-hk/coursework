import torch.nn as nn

from .blocks import ConvBlock, InceptionBlock, MLPBlock, ResBlockBottleneck, ResBlockPlain


class MyNetworkExample(nn.Module):
    def __init__(self, nf, block_type='mlp'):
        super(MyNetworkExample, self).__init__()
        if block_type == 'mlp':
            block = MLPBlock
            self.mlp = block(3*32*32, nf)
            self.fc = nn.Linear(nf, 10)
        else:
            raise Exception(f"Wrong type of block: {block_type}.Expected : mlp")

    def forward(self, x):
        output = self.mlp(x.view(x.size()[0], -1))
        output = self.fc(output)
        return output


class MyNetwork(nn.Module):
    def __init__(self, nf, block_type='conv', num_blocks=[1, 1, 1]):
        super(MyNetwork, self).__init__()

        self.block_type = block_type

        if self.block_type == 'conv':
            block = ConvBlock
            block_args = lambda x: (x, x, 3, 1, 1)
        elif self.block_type == 'resPlain':
            block = ResBlockPlain
            block_args = lambda x: (x,)
        elif self.block_type == 'resBottleneck':
            block = ResBlockBottleneck
            block_args = lambda x: (x, x//2)
        elif self.block_type == 'inception':
            block = InceptionBlock
            block_args = lambda x: (x, x)
        else:
            raise Exception(f"Wrong type of block: {block_type}")

        self.block1 = nn.Sequential(*[block(*block_args(nf)) for _ in range(num_blocks[0])])
        self.block2 = nn.Sequential(*[block(*block_args(nf*2)) for _ in range(num_blocks[1])])
        self.block3 = nn.Sequential(*[block(*block_args(nf*4)) for _ in range(num_blocks[2])])

        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = nf, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_features = nf)
        self.mxpl1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.cnn2 = nn.Conv2d(in_channels = nf, out_channels = nf*2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_features = nf*2)
        self.mxpl2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.cnn3 = nn.Conv2d(in_channels = nf*2, out_channels = nf*4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(num_features = nf*4)
        self.mxpl3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.adavpl = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.fl = nn.Flatten()
        self.ln = nn.Linear(nf*4, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        first_bl = self.block1(self.mxpl1(self.relu(self.bn1(self.cnn1(x)))))
        second_bl = self.block2(self.mxpl2(self.relu(self.bn2(self.cnn2(first_bl)))))
        third_bl = self.block3(self.mxpl3(self.relu(self.bn3(self.cnn3(second_bl)))))

        output = self.ln(self.fl(self.adavpl(third_bl)))
        return output
