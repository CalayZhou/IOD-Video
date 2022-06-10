from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from .base_dataset import BaseDataset

class IODVideo(BaseDataset):
    num_classes = 1
    def __init__(self, opt, mode):
        self.ROOT_DATASET_PATH = os.path.join(opt.root_dir, 'data/TLGDM')
        pkl_filename = 'TrueLeakedGas.pkl'
        super(IODVideo, self).__init__(opt, mode, self.ROOT_DATASET_PATH, pkl_filename)

    def imagefile(self, v, i):
        return os.path.join(self.ROOT_DATASET_PATH, 'Frames', v, '{:0>5}.png'.format(i))

