from functools import partial
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import math

from network.twod_models.common import TemporalPooling
from network.twod_models.temporal_modeling import temporal_modeling_module
# from network.deconv import deconv_layers
__all__ = ['resnet']

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_frames, stride=1, downsample=None, temporal_module=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.tam = temporal_module(duration=num_frames, channels=inplanes) \
            if temporal_module is not None else None

    def forward(self, x):
        identity = x
        if self.tam is not None:
            x = self.tam(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_frames, stride=1, downsample=None, temporal_module=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.tam = temporal_module(duration=num_frames, channels=inplanes) \
            if temporal_module is not None else None

    def forward(self, x):
        identity = x
        if self.tam is not None:
           x = self.tam(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity
        out = self.relu(out)

        return out


class IOD_TAM_ResNet(nn.Module):

    def __init__(self, depth,K):
        super(IOD_TAM_ResNet, self).__init__()

        pooling_method = 'max'
        self.pooling_method = pooling_method.lower()
        block = BasicBlock if depth < 50 else Bottleneck
        layers = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]}[depth]
        self.zero_init_residual=False
        self.depth = depth
        self.output_channel = 64
        self.num_frames = K# num_frames
        self.orig_num_frames =  K#num_frames
        self.num_classes = 1000  #num_classes
        without_t_stride = False if K>=8 else True
        self.without_t_stride = without_t_stride
        temporal_module = partial(temporal_modeling_module, name= 'TAM',
                              dw_conv=True,
                              blending_frames=3,
                              blending_method='sum')
        self.temporal_module = temporal_module

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if not self.without_t_stride:
            self.pool1 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if not self.without_t_stride:
            self.pool2 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if not self.without_t_stride:
            self.pool3 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        dropout = 0.5
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

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
        
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_frames, stride, downsample,
                            temporal_module=self.temporal_module))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.num_frames,
                                temporal_module=self.temporal_module))

        return nn.Sequential(*layers)

    def forward(self, x):
        # batch_size, c_t, h, w = x.shape
        # x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        batch_size, c, t, h, w = x.shape
        K =t
        x = x.view(batch_size * self.orig_num_frames, c ,t // self.orig_num_frames, h, w)
        x =x.squeeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        fp1 = self.maxpool(x)

        fp2 = self.layer1(fp1)
        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3 = self.layer2(fp2_d)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4 = self.layer3(fp3_d)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5 = self.layer4(fp4_d)
        
        # CSP deconv
        x1 = self.norm1(fp2)
        x2 = self.deconv2(fp3)
        x3 = self.deconv3(fp4)
        x4 = self.deconv4(fp5)

        bt, c,  h, w = x1.shape
        x1_split = x1.view(batch_size , c ,-1 , h, w).split(1,dim = 2)
        x2_split = x2.view(batch_size , c ,-1 , h, w).split(1,dim = 2)
        x3_split = x3.view(batch_size , c ,-1 , h, w).split(1,dim = 2)
        x4_split = x4.view(batch_size , c ,-1 , h, w).split(1,dim = 2)

        x_output = []
        for i  in range(K):
            map = torch.cat([x1_split[i].squeeze(2),x2_split[i//2].squeeze(2),x3_split[i//4].squeeze(2),x4_split[i//8].squeeze(2)],dim=1)
            # map = torch.cat([x1_split[i].squeeze(2),x2_split[i].squeeze(2),x3_split[i].squeeze(2),x4_split[i].squeeze(2)],dim=1)
            map = self.cat(map)
            map = self.cat_bn(map)
            map = self.cat_act(map)
            map = self.cat64(map)
            map = self.cat_bn64(map)
            x_output.append(self.cat_act(map))

        return x_output

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = ''
        if self.temporal_module is not None:
            param = signature(self.temporal_module).parameters
            temporal_module = str(param['name']).split("=")[-1][1:-1]
            blending_frames = str(param['blending_frames']).split("=")[-1]
            blending_method = str(param['blending_method']).split("=")[-1][1:-1]
            dw_conv = True if str(param['dw_conv']).split("=")[-1] == 'True' else False
            name += "{}-b{}-{}{}-".format(temporal_module, blending_frames,
                                         blending_method,
                                         "" if dw_conv else "-allc")
        name += 'resnet-{}'.format(self.depth)
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)

        return name

