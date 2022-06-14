# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from network.twod_models.base_module import *
import torch.nn.init as init
import math

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


class IOD_TDN_ResNet(nn.Module):

    def __init__(self, depth,K):
        super(IOD_TDN_ResNet, self).__init__()
        self.num_segments = K
        resnet_model = fbresnet50(num_segments=self.num_segments, pretrained=True)
        resnet_model1 = fbresnet50(num_segments=self.num_segments, pretrained=True)
        apha = 0.5
        belta = 0.5

        self.output_channel = 64
        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

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


        self.cat = nn.Conv2d(1024,256,kernel_size=3,stride=1,padding=1)
        self.cat64 = nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1)
        self.cat_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.cat_bn64 = nn.BatchNorm2d(64, momentum=0.01)
        self.cat_act = nn.ReLU(inplace=True)

        # implement conv1_5 and inflate weight
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 4,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                     nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 = nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = list(resnet_model.children())[8]
        self.apha = apha
        self.belta = belta


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

    def forward(self, inputpre):
        x_list = []
        for i in range(self.num_segments):
            x1, x2, x3, x4, x5 = inputpre[i + 0].clone(), inputpre[i + 1].clone(), inputpre[i + 2].clone(), inputpre[
                i + 3].clone(), inputpre[i + 4].clone()
            x_c5 = self.conv1_5(
                self.avg_diff(torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 1).view(-1, 12, x2.size()[2], x2.size()[3])))
            x_diff = self.maxpool_diff(1.0 / 1.0 * x_c5)

            temp_out_diff1 = x_diff
            x_diff = self.resnext_layer1(x_diff)

            x = self.conv1(x3)
            x = self.bn1(x)
            x = self.relu(x)
            # fusion layer1
            x = self.maxpool(x)
            temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
            x = self.apha * x + self.belta * temp_out_diff1
            # fusion layer2
            x = self.layer1_bak(x)
            x_diff = F.interpolate(x_diff, x.size()[2:])
            x = self.apha * x + self.belta * x_diff
            x_list.append(x)

        x0 = torch.cat(x_list, dim=0)# B*T C  H W
        x1 = self.layer2_bak(x0)
        x2 = self.layer3_bak(x1)
        x3 = self.layer4_bak(x2)

        x0 = x0.unsqueeze(2)
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        x3 = x3.unsqueeze(2)
        batch_size_new, c, t, h, w = x0.shape
        x0 = x0.view( -1 , c, self.num_segments  , h, w)
        batch_size_new, c, t, h, w = x1.shape
        x1 = x1.view( -1 , c, self.num_segments  , h, w)
        batch_size_new, c, t, h, w = x2.shape
        x2 = x2.view( -1 , c, self.num_segments  , h, w)
        batch_size_new, c, t, h, w = x3.shape
        x3 = x3.view( -1 , c, self.num_segments  , h, w)

        x0_split = x0.split(1,dim = 2)
        x1_split = x1.split(1,dim = 2)
        x2_split = x2.split(1,dim = 2)
        x3_split = x3.split(1,dim = 2)

        x0_split_deconv = [self.norm1(x0_split[i].squeeze(2)) for i in range(len(x0_split))]
        x1_split_deconv = [self.deconv2(x1_split[i].squeeze(2)) for i in range(len(x1_split))]
        x2_split_deconv = [self.deconv3(x2_split[i].squeeze(2)) for i in range(len(x2_split))]
        x3_split_deconv = [self.deconv4(x3_split[i].squeeze(2)) for i in range(len(x3_split))]

        x_output = []
        for i in range(self.num_segments):
            map = torch.cat([x0_split_deconv[i],x1_split_deconv[i], x2_split_deconv[i], x3_split_deconv[i]], dim=1)
            map = self.cat(map)
            map = self.cat_bn(map)
            map = self.cat_act(map)
            map = self.cat64(map)
            map = self.cat_bn64(map)
            x_output.append(self.cat_act(map))

        return x_output
