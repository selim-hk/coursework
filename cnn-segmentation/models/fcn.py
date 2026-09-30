import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG, make_layers

from utils import get_upsampling_weight

cfg = {'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


class FCN32(VGG):
    def __init__(self):
        super(FCN32, self).__init__(make_layers(cfg['vgg16']))

        self.numclass = 21

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()

        self.cnl_fc6 = nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = 7)
        self.cnl_fc7 = nn.Conv2d(in_channels = 4096, out_channels = 4096, kernel_size = 1)

        self.cnl_1 = nn.Conv2d(in_channels = 4096, out_channels = 21, kernel_size = 1)

        self.dcnl = nn.ConvTranspose2d(in_channels = 21, out_channels = 21, kernel_size = 64, stride = 32, bias = False)

        self._initialize_weights()

    def load_pretrained(self, pretrained_model):
        self.features = pretrained_model.features
        fc6 = pretrained_model.classifier[0]
        fc7 = pretrained_model.classifier[3]

        fc6_parameters = fc6.state_dict()
        fc7_parameters = fc7.state_dict()

        fc6_parameters["weight"] = fc6_parameters["weight"].view(4096,512,7,7)
        fc7_parameters["weight"] = fc7_parameters["weight"].view(4096,4096,1,1)

        self.cnl_fc6.load_state_dict(fc6_parameters)
        self.cnl_fc7.load_state_dict(fc7_parameters)

    def vgg_layer_forward(self, x, indices):
        output = x
        start_idx, end_idx = indices
        for idx in range(start_idx, end_idx):
            output = self.features[idx](output)
        return output

    def vgg_forward(self, x):
        out = {}
        layer_indices = [0, 5, 10, 17, 24, 31]
        for layer_num in range(len(layer_indices)-1):
            x = self.vgg_layer_forward(x, layer_indices[layer_num:layer_num+2])
            out[f'pool{layer_num+1}'] = x
        return out

    def forward(self, x):
        padded_x = F.pad(x, [100, 100, 100, 100], "constant", 0)
        vgg_features = self.vgg_forward(padded_x)
        vgg_pool5 = vgg_features['pool5'].detach()
        vgg_pool4 = vgg_features['pool4'].detach()
        vgg_pool3 = vgg_features['pool3'].detach()

        classified = self.cnl_1(self.relu(self.cnl_fc7(self.relu(self.cnl_fc6(vgg_pool5)))))

        out = self.dcnl(classified)

        h1 = x.shape[2]
        w1 = x.shape[3]

        out = out[:, :, 9:(h1 + 9), 9:(w1+9) ]

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


class FCN8(FCN32):
    def __init__(self):
        super(FCN8, self).__init__()

        self.numclass = 21

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d()

        self.cnl_5 = nn.Conv2d(in_channels = 4096, out_channels = 21, kernel_size = 1)
        self.cnl_4 = nn.Conv2d(in_channels = 512, out_channels = 21, kernel_size = 1)
        self.cnl_3 = nn.Conv2d(in_channels = 256, out_channels = 21, kernel_size = 1)

        self.dcnl1 = nn.ConvTranspose2d(in_channels = 21, out_channels = 21, kernel_size = 4,stride = 2, bias = False)
        self.dcnl2 = nn.ConvTranspose2d(in_channels = 21, out_channels = 21, kernel_size = 4,stride = 2, bias = False)
        self.dcnl3 = nn.ConvTranspose2d(in_channels = 21, out_channels = 21, kernel_size = 16,stride = 8, bias = False)

        self._initialize_weights()

    def forward(self, x):
        padded_x = F.pad(x, [100, 100, 100, 100], "constant", 0)
        vgg_features = self.vgg_forward(padded_x)
        vgg_pool5 = vgg_features['pool5'].detach()
        vgg_pool4 = vgg_features['pool4'].detach()
        vgg_pool3 = vgg_features['pool3'].detach()

        after_first_deconv = self.dcnl1(self.cnl_5(self.relu(self.cnl_fc7(self.relu(self.cnl_fc6(vgg_pool5))))))

        h1 = after_first_deconv.shape[2]
        w1 = after_first_deconv.shape[3]

        pool4_after_deconv = self.cnl_4(vgg_pool4 * 0.01)
        pool4_after_crop = pool4_after_deconv[:, :, 5:(h1+5), 5:(w1+5)]

        after_second_deconv = self.dcnl2(after_first_deconv + pool4_after_crop)

        h2 = after_second_deconv.shape[2]
        w2 = after_second_deconv.shape[3]

        pool3_after_dconv = self.cnl_3(vgg_pool3 * 0.01)
        pool3_after_crop = pool3_after_dconv[:, :, 9:(h2+9), 9:(w2+9)]

        after_third_deconv = self.dcnl3(after_second_deconv + pool3_after_crop)

        h = x.shape[2]
        w = x.shape[3]

        out = after_third_deconv[:, :, 31:(h+31), 31:(w + 31)]

        return out
