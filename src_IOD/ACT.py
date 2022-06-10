from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

from datasets.init_dataset import get_dataset

from opts import opts
from ACT_utils.ACT_utils import iou2d, pr_to_ap
from ACT_utils.ACT_build import load_frame_detections, BuildTubes

def frameAP(opt, print_info=True):
    redo = opt.redo
    th = opt.th
    split = 'val'
    model_name = opt.model_name
    Dataset = get_dataset(opt.dataset)
    dataset = Dataset(opt, split)

    inference_dirname = opt.inference_dir
    print('inference_dirname is ', inference_dirname)
    print('threshold is ', th)

    vlist = dataset._test_videos[opt.split - 1]

    # load per-frame detections
    frame_detections_file = os.path.join(inference_dirname, 'frame_detections.pkl')
    if os.path.isfile(frame_detections_file) and not redo:
        print('load previous linking results...')
        print('if you want to reproduce it, please add --redo')
        with open(frame_detections_file, 'rb') as fid:
            alldets = pickle.load(fid)
    else:
        alldets = load_frame_detections(opt, dataset, opt.K, vlist, inference_dirname)
        try:
            with open(frame_detections_file, 'wb') as fid:
                pickle.dump(alldets, fid, protocol=4)
        except:
            print("OverflowError: cannot serialize a bytes object larger than 4 GiB")

    results = {}
    # compute AP for each class
    for ilabel, label in enumerate(dataset.labels):
        # detections of this class
        detections = alldets[alldets[:, 2] == ilabel, :]

        # load ground-truth of this class
        gt = {}
        for iv, v in enumerate(vlist):
            tubes = dataset._gttubes[v]

            if ilabel not in tubes:
                continue

            for tube in tubes[ilabel]:
                for i in range(tube.shape[0]):
                    k = (iv, int(tube[i, 0]))
                    if k not in gt:
                        gt[k] = []
                    gt[k].append(tube[i, 1:5].tolist())

        for k in gt:
            gt[k] = np.array(gt[k])
        
        #print(detections.shape())
        #print(gt.shape())
        # pr will be an array containing precision-recall values
        pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
        pr[0, 0] = 1.0
        pr[0, 1] = 0.0
        fn = sum([g.shape[0] for g in gt.values()])  # false negatives
        fp = 0  # false positives
        tp = 0  # true positives

        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]
            ispositive = False

            if k in gt:
                ious = iou2d(gt[k], box)
                amax = np.argmax(ious)

                if ious[amax] >= th:
                    ispositive = True
                    gt[k] = np.delete(gt[k], amax, 0)

                    if gt[k].size == 0:
                        del gt[k]

            if ispositive:
                tp += 1
                fn -= 1
            else:
                fp += 1

            pr[i + 1, 0] = float(tp) / float(tp + fp)
            pr[i + 1, 1] = float(tp) / float(tp + fn)

        results[label] = pr

    # display results
    ap = 100 * np.array([pr_to_ap(results[label]) for label in dataset.labels])
    frameap_result = np.mean(ap)
    if print_info:
        log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
        log_file.write('\nTask_{} frameAP_{}\n'.format(model_name, th))
        print('Task_{} frameAP_{}\n'.format(model_name, th))
        log_file.write("\n{:20s} {:8.2f}\n\n".format("mAP", frameap_result))
        log_file.close()
        print("{:20s} {:8.2f}".format("mAP", frameap_result))

    return frameap_result

def frameAP_050_095(opt):
    ap = 0
    for i in range(10):
        opt.th = 0.5 + 0.05 * i
        ap += frameAP(opt, print_info=False)
    ap = ap / 10.0
    log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
    log_file.write('\nTask_{} FrameAP_0.50:0.95 \n'.format(opt.model_name))
    log_file.write("\n{:20s} {:8.2f}\n\n".format("mAP", ap))
    log_file.close()
    print('Task_{} FrameAP_0.50:0.95 \n'.format(opt.model_name))
    print("\n{:20s} {:8.2f}\n\n".format("mAP", ap))



if __name__ == "__main__":
    opt = opts().parse()
    if not os.path.exists(os.path.join(opt.root_dir, 'result')):
        os.system("mkdir -p '" + os.path.join(opt.root_dir, 'result') + "'")
    if opt.task == 'BuildTubes':
        BuildTubes(opt)
    elif opt.task == 'frameAP':
        frameAP(opt)
    elif opt.task == 'frameAP_all':
        frameAP_050_095(opt)
    else:
        raise NotImplementedError('Not implemented:' + opt.task)
