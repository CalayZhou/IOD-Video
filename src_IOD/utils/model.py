from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
# import torchvision.models as models
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
from network.STA_Framework import STA_Framework
from network.inflate_from_2d_model import inflate_from_2d_model


# transmit parameters to class STA_Framework
def create_model(arch, branch_info, head_conv, K):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    model = STA_Framework(arch, num_layers, branch_info, head_conv, K)
    return model


def load_model(model, model_path, optimizer=None, lr=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        # if ucf_pretrain:
        #     if k.startswith('branch.hm') or k.startswith('branch.mov'):
        #         continue
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    check_state_dict(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Resumed optimizer with start lr', lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if 'best' in checkpoint:
        best = checkpoint['best']
    else:
        best = 100
    if optimizer is not None:
        return model, optimizer, start_epoch, best
    else:
        return model

def save_model(path, model, optimizer=None, epoch=0, best=100):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'best': best,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def load_imagenet_pretrained_model(opt, model):
    
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'dla34': 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth',
        'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
        'res2net50_48w_2s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_48w_2s-afed724a.pth',
        'res2net50_14w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_14w_8s-6527dddc.pth',
        'res2net50_26w_6s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_6s-19041792.pth',
        'res2net50_26w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_8s-2c7c9f12.pth',
        'res2net101_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net101_26w_4s-02a759a1.pth',
        'fbresnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
        'fbresnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth'
    }    

    arch = opt.arch
    if arch == 'dla_34':
        print('load imagenet pretrained dla_34')
        model_url = model_urls['dla34']
        model_weights = model_zoo.load_url(model_url)


    elif arch == 'MSresnet_50':
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict_copy = pretrained_dict.copy()
        new_state_dict = model.state_dict()
        pretrained_dict = OrderedDict()
        for key, value in pretrained_dict_copy.items():
            pretrained_dict['backbone.' + key] = value
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k: v})
            #                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
        #check_state_dict(model.state_dict(), pretrained_dict)
        return model

    elif arch == 'TINresnet_50' or arch == 'TSMresnet_50':
        # url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        state_dict = model_zoo.load_url(model_urls['resnet50'] )
        state_dict_copy = state_dict.copy()
        state_dict = OrderedDict()
        for key, value in state_dict_copy.items():
            state_dict['backbone.base_model.' + key] = value
        model.load_state_dict(state_dict, strict=False)
        # check_state_dict(model.state_dict(), state_dict)
        return model

    elif arch == 'TDNresnet_50':
        '''
        the pretrained model is loaded in /network/twod_models/base_module.py
        '''
        return model

    elif arch == 'TAMresnet_50':
        new_model_state_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['resnet50'] , map_location='cpu', progress=True)
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        state_dict_copy = state_dict.copy()
        state_dict = OrderedDict()
        for key, value in state_dict_copy.items():
            state_dict['backbone.' + key] = value
        model.load_state_dict(state_dict, strict=False)
        return model

    elif arch == 'TEAresnet_50':
        new_model_state_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['res2net50_26w_4s'] , map_location='cpu', progress=True)
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        state_dict_copy = state_dict.copy()
        state_dict = OrderedDict()
        for key, value in state_dict_copy.items():
            state_dict['backbone.' + key] = value
        model.load_state_dict(state_dict, strict=False)
        #check_state_dict(model.state_dict(), state_dict)
        return model

    elif arch == 'S3Dresnet_50':
        new_model_state_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['resnet50'] , map_location='cpu', progress=True)
        state_d = inflate_from_2d_model(state_dict, new_model_state_dict,skipped_keys=['fc'])
        model.load_state_dict(state_d, strict=False)
        return model

    elif arch == 'I3Dresnet_50':
        new_model_state_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['resnet50'] , map_location='cpu', progress=True)
        state_d = inflate_from_2d_model(state_dict, new_model_state_dict,skipped_keys=['fc'])
        model.load_state_dict(state_d, strict=False)
        return model

    elif arch.startswith('resnet'):
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        assert num_layers in (18, 34, 50, 101, 152)
        arch = arch[:arch.find('_')] if '_' in arch else arch

        print('load imagenet pretrained ', arch)
        url = model_urls['resnet{}'.format(num_layers)]
        model_weights = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))

    else:
        raise NotImplementedError
    new_state_dict = {}
    for key, value in model_weights.items():
        new_key = 'backbone.base.' + key
        new_state_dict[new_key] = value
    if opt.print_log:
        check_state_dict(model.state_dict(), new_state_dict)
        print('check done!')
    model.load_state_dict(new_state_dict, strict=False)

    return model


def check_state_dict(load_dict, new_dict):
    # check loaded parameters and created model parameters
    for k in new_dict:
        if k in load_dict:
            if new_dict[k].shape != load_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(
                          k, load_dict[k].shape, new_dict[k].shape))
                new_dict[k] = load_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in load_dict:
        if not (k in new_dict):
            print('No param {}.'.format(k))
            new_dict[k] = load_dict[k]
