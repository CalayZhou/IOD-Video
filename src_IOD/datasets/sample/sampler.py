from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random
import numpy as np
import cv2
import torch.utils.data as data
from utils.gaussian_hm import gaussian_radius, draw_umich_gaussian
from ACT_utils.ACT_aug import apply_distort, apply_expand, crop_image

class Sampler(data.Dataset):
    def __getitem__(self, id):
        v, frame = self._indices[id]
        frame_reorder = frame
        K = self.K
        num_classes = self.num_classes
        input_h = self._resize_height
        input_w = self._resize_width
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        # read images
        if self.opt.arch == 'TDNresnet_50':
            images = []
            for i in range(K+4):
                frame_i = frame + i - 2
                video_len = self._nframes[v]
                if(frame_i>video_len):
                    frame_i = video_len
                if(frame_i<1):
                    frame_i = 1
                image =cv2.imread(self.imagefile(v, frame_i )).astype(np.float32)
                images.append(image)
        else:
            images = []
            for i in range(K):
                frame_i = frame + i
                video_len = self._nframes[v]
                if(frame_i>video_len):
                    frame_i = video_len
                if(frame_i<1):
                    frame_i = 1
                image =cv2.imread(self.imagefile(v, frame_i )).astype(np.float32)
                images.append(image)
        cv2.setNumThreads(0)

        if self.opt.arch == 'TDNresnet_50':
            data = [np.empty((3 , self._resize_height, self._resize_width), dtype=np.float32) for i in range(K+4)]
        else:
            data = [np.empty((3 , self._resize_height, self._resize_width), dtype=np.float32) for i in range(K)]

        '''
        image pre process
        '''
        if self.mode == 'train':
            do_mirror = random.getrandbits(1) == 1
            # filp the image
            if do_mirror:
                images = [im[:, ::-1, :] for im in images]
            h, w = self._resolution[v]
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue
                    # assert frame + K - 1 in t[:, 0]
                    # copy otherwise it will change the gt of the dataset also
                    t = t.copy()
                    if do_mirror:
                        # filp the gt bbox
                        xmin = w - t[:, 3]
                        t[:, 3] = w - t[:, 1]
                        t[:, 1] = xmin
                    boxes = []
                    #avoid frame_i out of the video length
                    for frame_i  in range(frame,frame+K,1):
                        if(frame_i>=t.shape[0]):
                            frame_i = t.shape[0]
                        boxes.append(t[frame_i-1,1:5])
                    boxes = np.array(boxes)

                    assert boxes.shape[0] == K
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    # gt_bbox[ilabel] ---> a list of numpy array, each one is K, x1, x2, y1, y2
                    gt_bbox[ilabel].append(boxes)

            # apply data augmentation
            images = apply_distort(images, self.distort_param)
            images, gt_bbox,expand_ratio = apply_expand(images, gt_bbox, self.expand_param, self._mean_values)
            images, gt_bbox,crop_areas = crop_image(images, gt_bbox, self.batch_samplers)

        else:
            # no data augmentation or flip when validation
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue
                    assert frame + K - 1 in t[:, 0]
                    t = t.copy()
                    boxes = []
                    for frame_i  in range(frame,frame+K,1):
                        if(frame_i>=t.shape[0]):#1109
                            frame_i = t.shape[0]
                        boxes.append(t[frame_i-1,1:5])
                    boxes = np.array(boxes)
                    
                    assert boxes.shape[0] == K
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    gt_bbox[ilabel].append(boxes)

        original_h, original_w = images[0].shape[:2]
        # resize the original img and it's GT bbox
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                gt_bbox[ilabel][itube][:, 0] = gt_bbox[ilabel][itube][:, 0] / original_w * output_w
                gt_bbox[ilabel][itube][:, 1] = gt_bbox[ilabel][itube][:, 1] / original_h * output_h
                gt_bbox[ilabel][itube][:, 2] = gt_bbox[ilabel][itube][:, 2] / original_w * output_w
                gt_bbox[ilabel][itube][:, 3] = gt_bbox[ilabel][itube][:, 3] / original_h * output_h
        images = [cv2.resize(im, (input_w, input_h), interpolation=cv2.INTER_LINEAR) for im in images]
        # transpose image channel and normalize
        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (1, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (1, 1, 1))
        
        if self.opt.arch == 'TDNresnet_50':
            K_i = K+4
        else:
            K_i = K
        for i in range(K_i):
            data[i][ 0 : 3, :, :] = np.transpose(images[i], (2, 0, 1))
            data[i] = ((data[i] / 255.) - mean) / std

        # draw ground truth
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, K * 2), dtype=np.float32)
        mov = np.zeros((self.max_objs, K * 2), dtype=np.float32)
        centerKpoints = np.zeros((self.max_objs, K * 2), dtype=np.float32)
        index = np.zeros((self.max_objs), dtype=np.int64)
        index_all = np.zeros((self.max_objs, K * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                key = K // 2
                # key frame's bbox height and width （both on the feature map）
                key_h, key_w = gt_bbox[ilabel][itube][key, 3] - gt_bbox[ilabel][itube][key, 1], gt_bbox[ilabel][itube][key, 2] - gt_bbox[ilabel][itube][key, 0]
                # create gaussian heatmap
                radius = gaussian_radius((math.ceil(key_h), math.ceil(key_w)))
                radius = max(0, int(radius))
                # ground truth bbox's center in key frame
                center = np.array([(gt_bbox[ilabel][itube][key, 0] + gt_bbox[ilabel][itube][key, 2]) / 2, (gt_bbox[ilabel][itube][key, 1] + gt_bbox[ilabel][itube][key, 3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                assert 0 <= center_int[0] and center_int[0] <= output_w and 0 <= center_int[1] and center_int[1] <= output_h
                # draw ground truth gaussian heatmap at each center location
                draw_umich_gaussian(hm[ilabel], center_int, radius)
                for i in range(K):
                    center_all = np.array([(gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2,  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2], dtype=np.float32)
                    center_all_int = center_all.astype(np.int32)
                    # wh is ground truth bbox's height and width in i_th frame
                    wh[num_objs, i * 2: i * 2 + 2] = 1. * (gt_bbox[ilabel][itube][i, 2] - gt_bbox[ilabel][itube][i, 0]), 1. * (gt_bbox[ilabel][itube][i, 3] - gt_bbox[ilabel][itube][i, 1])
                    # mov is ground truth movement from i_th frame to key frame
                    mov[num_objs, i * 2: i * 2 + 2] = (gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2 - \
                       center_int[0],  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2 - center_int[1]
                    # centerKpoints is the box center of ground truth for all K frames
                    centerKpoints[num_objs, i * 2: i * 2 + 2] = (gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2  \
                        ,  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2 
                    # index_all are all frame's bbox center position
                    index_all[num_objs, i * 2: i * 2 + 2] = center_all_int[1] * output_w + center_all_int[0], center_all_int[1] * output_w + center_all_int[0]
                # index is key frame's boox center position
                index[num_objs] = center_int[1] * output_w + center_int[0]
                # mask indicate how many objects in this tube
                mask[num_objs] = 1
                num_objs = num_objs + 1

        result = {'input': data, 'hm': hm, 'mov': mov,'centerKpoints': centerKpoints, 'wh': wh, 'mask': mask, 'index': index, 'index_all': index_all}
        return result
