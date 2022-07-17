from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import pickle
import sys

from tiny_opt import opts
from vis_dataset import VisualizationDataset
from build import build_tubes
from vis_utils import pkl_decode, vis_bbox, rgb2avi, video2frames, rgb2gif
sys.path.append("..")
from detector.normal_det import Detector
from ACT_utils.ACT_utils import nms_tubelets

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process):
        self.pre_process = pre_process
        self.opt = opt
        self.nframes = dataset._nframes
        self.imagefile = dataset.imagefile

        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
        for i in range(1, 1 + self.nframes - self.opt.K + 1):
            if not os.path.exists(self.outfile(i)):
                self.indices.append(i)
        self.last_frame = -1
        self.h, self.w, _ = cv2.imread(self.imagefile(1)).shape

    def __getitem__(self, index):
        frame = self.indices[index]

        # if there is a new video
        if frame == self.last_frame + 1:
            video_tag = 1
        else:
            video_tag = 0

        self.last_frame = frame

        images = [cv2.imread(self.imagefile(frame + i)).astype(np.float32) for i in range(self.opt.K)]
        images = self.pre_process(images,self.opt.K)

        outfile = self.outfile(frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'meta': {'height': self.h, 'width': self.w, 'output_height': self.output_h, 'output_width': self.output_w}, 'video_tag': video_tag}

    def outfile(self, i):
        return os.path.join(self.opt.inference_dir, 'VideoFrames', "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)


def demo_inference(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # torch.backends.cudnn.benchmark = True

    dataset = VisualizationDataset(opt)
    detector = Detector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process)#, detector.pre_process_single_frame)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False)

    print('inference begin!', flush=True)
    for iter, data in enumerate(data_loader):

        outfile = data['outfile']

        detections = detector.run(data)

        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)#{Class1: N 4*K+1}


def simple_inference(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # torch.backends.cudnn.benchmark = True
    dataset = VisualizationDataset(opt)
    detector = Detector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process)#, detector.pre_process_single_frame)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False)
    VDets = {}
    K = opt.K
    label_name, tube_id = 'LeakedGas',0
    print('inference begin!', flush=True)
    for iter, data in enumerate(data_loader):
        detections = detector.run(data)
        VDets[iter+1] = nms_tubelets(detections[0][1], 0.6, top_k=1)
    print('inference finish!', flush=True)

    # load detected tubelets
    bbox_dict = {}
    for frame in range(1, dataset._nframes + 2 - K):
        bbox_dict[frame] = []
        #simply retain only one box
        x1, y1, x2, y2,frame_score = VDets[frame][0,0], VDets[frame][0,1], VDets[frame][0,2], VDets[frame][0,3], VDets[frame][0,-1]

        bbox_dict[frame].append([x1, y1, x2, y2, frame_score, label_name, tube_id])

    #bbox_dict {frame:[x1, y1, x2, y2, frame_score, label_name, tube_id]}
    return  bbox_dict

def det():
    opt = opts().parse()
    os.system("rm -rf " +os.path.join(opt.inference_dir, 'VideoFrames') )#+ "/*")
    os.system("rm -rf tmp")
    os.system("mkdir -p '" + os.path.join(opt.inference_dir, 'VideoFrames') + "'")

    print('inference '+opt.vname+' start!')

    #extract frames from video
    video2frames(opt)

    #process in a simple way
    if opt.SimpleFrameProcess:
        bbox_dict_frame = simple_inference(opt)
        vis_bbox(opt, os.path.join(opt.inference_dir, 'VideoFrames'), bbox_dict_frame, opt.instance_level)
        rgb2avi(opt.inference_dir,opt.vname+'_simpleframe_result.avi')

    #demo by linking video tubes
    else:
        #generate pkl for each frame
        demo_inference(opt)
        #link frame bboxes to build tubes
        build_tubes(opt)
        #decode tubes
        bbox_dict_video = pkl_decode(opt)
        #draw boxes
        vis_bbox(opt, os.path.join(opt.inference_dir, 'VideoFrames'), bbox_dict_video, opt.instance_level)
        #write video
        rgb2avi(opt.inference_dir,opt.vname+'_videotubes_result.avi')


    if opt.save_gif:
        rgb2gif(opt)

    os.system("rm -rf tmp")
    os.system("rm -rf " + opt.inference_dir + "/rgb")

    print('Finish!', flush=True)


if __name__ == '__main__':
    det()
