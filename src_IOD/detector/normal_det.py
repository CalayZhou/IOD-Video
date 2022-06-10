from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch
from utils.model import create_model, load_model
from utils.data_parallel import DataParallel
from .decode import decode



'''
Detector define the whole end-to-end model

opt.arch 'resnet_50|I3Dresnet_50|S3Dresnet_50|MSresnet_50|TDNresnet_50|TAMresnet|TSMresnet_50|TINresnet_50|TEAresnet_50'

opt.branch_info = {'hm': opt.num_classes,
                   'mov': 2 * opt.K,
                   'wh': 2 * opt.K}
            
'--set_head_conv', type=int, default=-1,
             help='conv layer channels for output head'
                  'default setting is 256 for dla and 256 for resnet(except for wh branch) ')
'''

class Detector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        self.rgb_model = None
        if opt.rgb_model != '':
            print('create rgb model')
            self.rgb_model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K)
            self.rgb_model = load_model(self.rgb_model, opt.rgb_model)
            self.rgb_model = DataParallel(
                self.rgb_model, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.rgb_model.eval()
        self.num_classes = opt.num_classes
        self.opt = opt


    def pre_process(self, images):

        K = self.opt.K
        images = [cv2.resize(im, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR) for im in images]

        data = [np.empty((3 , self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K)]

        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (1, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (1, 1, 1))

        for i in range(K):
            data[i][0 : 3, :, :] = np.transpose(images[i], (2, 0, 1))
            # normalize
            data[i] = ((data[i] / 255.) - mean) / std
        return data

    def process(self, images):
        with torch.no_grad():
            if self.rgb_model is not None:
                rgb_output  = self.rgb_model(images)
                hm = rgb_output[0]['hm'].sigmoid_()
                wh = rgb_output[0]['wh']
                STAoffset = rgb_output[0]['STA_offset']
                mov = rgb_output[0]['mov']

            detections = decode(hm, wh,mov, STAoffset, self.opt, N=self.opt.N, K=self.opt.K)
            return detections

    def post_process(self, detections, height, width, output_height, output_width, K):
        detections = detections.detach().cpu().numpy()

        results = []
        for i in range(detections.shape[0]):
            top_preds = {}
            for j in range((detections.shape[2] - 2) // 2):
                # tailor bbox to prevent out of bounds
                detections[i, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[i, :, 2 * j] / output_width * width))
                detections[i, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[i, :, 2 * j + 1] / output_height * height))
            classes = detections[i, :, -1]
            # gather bbox for each class
            for c in range(self.opt.num_classes):
                inds = (classes == c)
                top_preds[c + 1] = detections[i, inds, :4 * K + 1].astype(np.float32)
            results.append(top_preds)
        return results

    def run(self, data):

        images = None

        if self.rgb_model is not None:
            images = data['images']
            for i in range(len(images)):
                images[i] = images[i].to(self.opt.device)

        meta = data['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}

        detections = self.process(images)

        detections = self.post_process(detections, meta['height'], meta['width'],
                                       meta['output_height'], meta['output_width'],
                                        self.opt.K)

        return detections
