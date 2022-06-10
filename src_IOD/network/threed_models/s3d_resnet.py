import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torch.nn.init as init
from network.inflate_from_2d_model import inflate_from_2d_model

__all__ = ['s3d_resnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def BasicConv3d(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0),
                bias=False, dw_t_conv=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=bias,
                     groups=in_planes if dw_t_conv else 1)


class STBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1, 1), padding=0, downsample=None,
                 dw_t_conv=False):
        super(STBasicBlock, self).__init__()

        self.conv1 = BasicConv3d(inplanes, planes, kernel_size=(1, 3, 3),
                                 stride=(1, stride[1], stride[2]), padding=(0, padding, padding),
                                 bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_t = BasicConv3d(planes, planes, kernel_size=(3, 1, 1),
                                   stride=(stride[0], 1, 1), padding=(padding, 0, 0), bias=False,
                                   dw_t_conv=dw_t_conv)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.relu1_t = nn.ReLU(inplace=True)
        self.conv2 = BasicConv3d(planes, planes, kernel_size=(1, 3, 3),
                                 stride=(1, 1, 1), padding=(0, padding, padding),
                                 bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_t = BasicConv3d(planes, planes, kernel_size=(3, 1, 1),
                                   stride=(1, 1, 1), padding=(padding, 0, 0), bias=False,
                                   dw_t_conv=dw_t_conv)
        self.bn2_t = nn.BatchNorm3d(planes)
        self.relu2_t = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu1_t(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2_t(out)

        return out


class STBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1, 1), padding=0, downsample=None,
                 dw_t_conv=False):
        super(STBottleneck, self).__init__()
        self.conv1 = BasicConv3d(inplanes, planes, kernel_size=(1, 1, 1),
                                 stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = BasicConv3d(planes, planes, kernel_size=(1, 3, 3),
                                 stride=(1, stride[1], stride[2]), padding=(0, padding, padding),
                                 bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_t = BasicConv3d(planes, planes, kernel_size=(3, 1, 1),
                                   stride=(stride[0], 1, 1), padding=(padding, 0, 0), bias=False,
                                   dw_t_conv=dw_t_conv)
        self.bn2_t = nn.BatchNorm3d(planes)
        self.relu2_t = nn.ReLU(inplace=True)
        self.conv3 = BasicConv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1),
                                 stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu2_t(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class IOD_S3D_ResNet(nn.Module):
    def __init__(self, depth,K):
        super(IOD_S3D_ResNet, self).__init__()
        layers = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]}[depth]
        block = STBasicBlock if depth < 50 else STBottleneck
        self.num_classes=1000
        self.output_channel = 64
        self.dropout_parameter=0.5
        self.zero_init_residual=False
        self.dw_t_conv = False
        self.without_t_stride = False
        self.depth = depth
        self.inplanes = 64
        self.t_s = 1 if self.without_t_stride else 2
        self.conv1 = BasicConv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3),
                                 bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.dropout = nn.Dropout(self.dropout_parameter)
        # self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, STBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, STBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        #CSP deconv
        self.expansion = 4
        self.norm1 = L2Norm(64*self.expansion,10)
        self.deconv2 = nn.Sequential(L2Norm(128*self.expansion,10),
                           nn.ConvTranspose2d(128*self.expansion, 256, kernel_size=4, stride=2, padding=1,bias = True))
        self.deconv3 = nn.Sequential(L2Norm(256*self.expansion,10),
                          nn.ConvTranspose2d(256*self.expansion, 256, kernel_size=4, stride=4, padding=0,bias=True))
        self.deconv4 = nn.Sequential(L2Norm(512*self.expansion,10),
                                    nn.ConvTranspose2d(512*self.expansion, 512, kernel_size=4, stride=2, padding=1,bias = True),
                                    L2Norm(512,10),
                                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4, padding=0,bias = True))



        self.cat = nn.Conv2d(256*4,256,kernel_size=3,stride=1,padding=1)
        self.cat64 = nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1)
        self.cat_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.cat_bn64 = nn.BatchNorm2d(64, momentum=0.01)
        self.cat_act = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = 's3d-resnet-{}'.format(self.depth)
        if self.dw_t_conv:
            name += '-dw-t-conv'
        if not self.without_t_stride:
            name += '-ts'
        return name

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BasicConv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1),
                            stride=(self.t_s if stride == 2 else 1, stride, stride)),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=(self.t_s if stride == 2 else 1, stride, stride),
                            padding=1, downsample=downsample, dw_t_conv=self.dw_t_conv))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=1, dw_t_conv=self.dw_t_conv))

        return nn.Sequential(*layers)

    def forward(self, x):
        K = list(x.size())[2]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)# 2 256 72 72
        x2 = self.layer2(x1)# 2 512 36 36
        x3 = self.layer3(x2)# 2 1024 18 18
        x4 = self.layer4(x3)# 2 1024 18 18


        x1_split = x1.split(1,dim = 2)
        x2_split = x2.split(1,dim = 2)
        x3_split = x3.split(1,dim = 2)
        x4_split = x4.split(1,dim = 2)

        x1_split_deconv = [self.norm1(x1_split[i].squeeze(2)) for i in range(len(x1_split))]
        x2_split_deconv = [self.deconv2(x2_split[i].squeeze(2)) for i in range(len(x2_split))]
        x3_split_deconv = [self.deconv3(x3_split[i].squeeze(2)) for i in range(len(x3_split))]
        x4_split_deconv = [self.deconv4(x4_split[i].squeeze(2)) for i in range(len(x4_split))]


        x_output = []
        for i  in range(K):
            map = torch.cat([x1_split_deconv[i],x2_split_deconv[i//2],x3_split_deconv[i//4],x4_split_deconv[i//8]],dim=1)
            map = self.cat(map)
            map = self.cat_bn(map)
            map = self.cat_act(map)
            map = self.cat64(map)
            map = self.cat_bn64(map)
            x_output.append(self.cat_act(map))

        return x_output



