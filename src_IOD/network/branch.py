from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
BN_MOMENTUM = 0.1

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class IOD_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(IOD_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv
        
        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))    
        fill_fc_weights(self.wh)

        self.sta = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Tanh())
        fill_fc_weights(self.sta)

    def forward(self, input_chunk_pre):
        output = {}
        output_wh = []
        output_STA_offset = []

        for feature in input_chunk_pre:# list [ (2 64 72 72),......]
            output_wh.append(self.wh(feature))#list [ (2 2 72 72),......]
            output_STA_offset.append(self.sta(feature))


        input_chunk = torch.cat(input_chunk_pre, dim=1)#2 192 72 72
        output_wh = torch.cat(output_wh, dim=1)#2 2*k 72 72
        output_STA_offset = torch.cat(output_STA_offset, dim=1)

        output['hm'] = self.hm(input_chunk)
        output['mov'] = self.mov(input_chunk)
        output['STA_offset'] = output_STA_offset
        output['wh'] = output_wh
        return output






