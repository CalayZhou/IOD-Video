from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basical experiment settings
        self.parser.add_argument('--simple_th', type=float, default=0.0,
                                 help='min score for visualize frame in the simple process')
        self.parser.add_argument('--tube_vis_th', type=float, default=0.12,
                                 help='min score for visualize tube')
        self.parser.add_argument('--frame_vis_th', type=float, default=0.015,
                             help='min score for visualize individual frame')
        self.parser.add_argument('--DATA_ROOT', default='./demo_video',
                                 help='dataset root path')
        self.parser.add_argument('--inference_dir', default='./demo_video',
                                 help='vis inference_dir')
        self.parser.add_argument('--vname', default='001_wild_static_vague.avi',
                                 help='video name')
        self.parser.add_argument('--rgb_model', default='../../experiment/result_model/TEA_STA_K8S3/TEA_STA_K8S3_model_last.pth',
                                 help='path to rgb model')
        self.parser.add_argument('--instance_level', action='store_true',
                                 help='draw instance_level action bbox in different color')

        # model seeting
        self.parser.add_argument('--arch', default='TEAresnet_50',
                                 help='model architecture. Currently tested')
        self.parser.add_argument('--head_conv', type=int, default=256,
                                 help='conv layer channels for output head'
                                      'default setting is 256 ')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')
        self.parser.add_argument('--K', type=int, default=8,
                                 help='length of action tube')
        self.parser.add_argument('--num_classes', type=int, default=1,
                                 help='1 num_classes for TLGDM')
        self.parser.add_argument('--loss_option', default='STAloss',
                                 help='MOV or STAloss, MOV is the original implementation')

        # dataset seetings
        self.parser.add_argument('--resize_height', type=int, default=288,
                                 help='input image height')
        self.parser.add_argument('--resize_width', type=int, default=288,
                                 help='input image width')

        # inference settings
        self.parser.add_argument('--N', type=int, default=10,
                                 help='max number of output objects.')
        self.parser.add_argument('--IMAGE_ROOT', default='./',
                                 help='dataset root path')
        self.parser.add_argument('--save_gif', action='store_true',
                                 help='save uncompressed GIF')
        self.parser.add_argument('--SimpleFrameProcess', type=int, default=1,
                                 help='simple process in the frame level')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # if opt.flow_model != '':
        #     opt.pre_extracted_brox_flow = True
        opt.mean = [0.40789654, 0.44719302, 0.47026115]
        opt.std = [0.28863828, 0.27408164, 0.27809835]

        opt.offset_h_ratio = 1
        opt.offset_w_ratio = 1

        opt.gpus = [0]
        # opt.vname = opt.vname.split('_')[1] + '/' + opt.vname
        opt.branch_info = {'hm': opt.num_classes,
                           'mov': 2 * opt.K,
                           'wh': 2 * opt.K}

        opt.chunk_sizes = [1]#batch=1 only one sample

        return opt
