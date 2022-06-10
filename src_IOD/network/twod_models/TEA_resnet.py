# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 17:38:33
# modified from res2net

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import math
import torch.nn.init as init

__all__ = ['Res2Net', 'res2net50']


model_urls = {
    # 'res2net50_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_4s-06e79181.pth',
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net101_26w_4s-02a759a1.pth'
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

# class Identity(nn.Module):
#     r"""A placeholder identity operator that is argument-insensitive.
#     Args:
#         args: any argument (unused)
#         kwargs: any keyword argument (unused)
#     Examples::
#         >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
#         >>> input = torch.randn(128, 20)
#         >>> output = m(input)
#         >>> print(output.size())
#         torch.Size([128, 20])
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(Identity, self).__init__()
#
#     def forward(self, input: Tensor) -> Tensor:
#         return input


class MEModule(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment=8):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        # self.identity = nn.Identity()
        # self.identity = Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1)  # n, t-1, c//r, h, w
        
        # motion fea = t+1_fea - t_fea
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
        diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
        y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        y = y - 0.5
        output = x + x * y.expand_as(x)
        return output

class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """

    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels  #26
        self.n_segment = n_segment
        self.fold_div = n_div #2
        self.fold = self.input_channels // self.fold_div #13
        self.conv = nn.Conv1d(
            2*self.fold, 2*self.fold,
            kernel_size=3, padding=1, groups=2*self.fold,
            bias=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True # (26,1,3)
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        # shift by conv
        # import pdb; pdb.set_trace()
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch*h*w, c, self.n_segment)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x

class Bottle2neckShift(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, n_segment = 8,stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neckShift, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.n_segment = n_segment

        self.me = MEModule(width*scale, reduction=16, n_segment=self.n_segment)#8)

        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        shifts = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride,
                padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
            shifts.append(ShiftModule(width, n_segment=self.n_segment, n_div=2, mode='fixed'))#n_segment=8, n_div=2, mode='fixed'))
        shifts.append(ShiftModule(width, n_segment=self.n_segment, n_div=2, mode='shift'))#8

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.shifts = nn.ModuleList(shifts)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion,
                   kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        # import pdb; pdb.set_trace()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.me(out)

        spx = torch.split(out, self.width, 1)  # 4*(nt, c/4, h, w)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.shifts[i](sp)
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        last_sp = spx[self.nums]
        last_sp = self.shifts[self.nums](last_sp)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, last_sp), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(last_sp)), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        # import pdb; pdb.set_trace()
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IOD_TEA_Res2Net(nn.Module):
    #model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4, **kwargs)
    def __init__(self, depth,n_segment):
        super(IOD_TEA_Res2Net, self).__init__()

        block = Bottle2neckShift
        layers = [3, 4, 6, 3]
        baseWidth = 26
        scale = 4
        num_classes = 1000
        self.n_segment =n_segment
        self.output_channel = 64
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,128, layers[1], stride=2)
        self.layer3 = self._make_layer(block,256, layers[2], stride=2)
        self.layer4 = self._make_layer(block,512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        # self.cat = nn.Conv2d(256*3,256,kernel_size=3,stride=1,padding=1)
        # self.conv_output = nn.Conv2d(self.output_channel , self.output_channel * 6, kernel_size=3, stride=1, padding=1)
        self.cat = nn.Conv2d(1024,256,kernel_size=3,stride=1,padding=1)
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

    def _make_layer(self, block,planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
            stype='stage', baseWidth = self.baseWidth, scale=self.scale,n_segment=self.n_segment))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale,n_segment=self.n_segment))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        batch_size, c, t, h, w = x.shape
        self.n_segment = t
        x = x.view(batch_size * self.n_segment, c ,t // self.n_segment, h, w)
        x =x.squeeze(2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        x3 = x3.unsqueeze(2)
        x4 = x4.unsqueeze(2)

        batch_size_new, c, t, h, w = x1.shape
        batch_size = batch_size_new//self.n_segment
        x1 = x1.view(batch_size , c ,-1 , h, w)
        batch_size_new, c, t, h, w = x2.shape
        x2 = x2.view(batch_size , c ,-1 , h, w)
        batch_size_new, c, t, h, w = x3.shape
        x3 = x3.view(batch_size , c ,-1 , h, w)
        batch_size_new, c, t, h, w = x4.shape
        x4 = x4.view(batch_size , c ,-1 , h, w)

        x1_split = x1.split(1,dim = 2)
        x2_split = x2.split(1,dim = 2)
        x3_split = x3.split(1,dim = 2)
        x4_split = x4.split(1,dim = 2)

        x1_split_deconv = [self.norm1(x1_split[i].squeeze(2)) for i in range(len(x1_split))]
        x2_split_deconv = [self.deconv2(x2_split[i].squeeze(2)) for i in range(len(x2_split))]
        x3_split_deconv = [self.deconv3(x3_split[i].squeeze(2)) for i in range(len(x3_split))]
        x4_split_deconv = [self.deconv4(x4_split[i].squeeze(2)) for i in range(len(x4_split))]

        x_output = []
        for i  in range(self.n_segment):
            map = torch.cat([x1_split_deconv[i],x2_split_deconv[i],x3_split_deconv[i],x4_split_deconv[i]],dim=1)
            map = self.cat(map)
            map = self.cat_bn(map)
            map = self.cat_act(map)
            map = self.cat64(map)
            map = self.cat_bn64(map)
            x_output.append(self.cat_act(map))
        return x_output

