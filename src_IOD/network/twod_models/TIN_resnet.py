from torch import nn
from network.resnet import IOD_ResNet
from network.twod_models.basic_ops import ConsensusModule
# from network.twod_models.transforms import *
from torch.nn.init import normal_, constant_
import torchvision
import torch.utils.model_zoo as model_zoo


class IOD_TIN_ResNet(nn.Module):
    def __init__(self, depth,K):
        super(IOD_TIN_ResNet, self).__init__()
        consensus_type = 'avg'
        new_length = None
        # num_segments = 8
        num_class = 1
        before_softmax = True
        dropout = 0.8
        img_feature_dim = 256
        partial_bn = True
        is_shift = False
        shift_div = 8
        shift_place = 'blockres'
        fc_lr5 = False
        tin = True
        non_local = False
        temporal_pool = False
        base_model = 'resnet50'
        self.K = K
        self.output_channel = 64
        self.modality = 'RGB'
        self.num_segments = K
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        # self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        # self.pretrain = pretrain

        self.is_shift = is_shift
        self.shift_div = shift_div
        self.shift_place = shift_place


        self.tin = tin

        self.base_model_name = base_model
        self.fc_lr5 = fc_lr5
        self.temporal_pool = temporal_pool
        self.non_local = non_local

        if new_length is None:
            self.new_length = 1 if self.modality == "RGB" else 5
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)

        # feature_dim = self._prepare_tsn(num_class)

        # if self.modality == 'RGBDiff':
        #     print("Converting the ImageNet model to RGB+Diff init model")
        #     self.base_model = self._construct_diff_model(self.base_model)
        #     print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        # self._enable_pbn = partial_bn
        if partial_bn:
            self._enable_pbn = True
        # Override the default train() to freeze the BN parameters
        # count = 0
        # if self._enable_pbn:
        #     print("Freezing BatchNorm2D except the first one.")
        #     for m in self.base_model.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             count += 1
        #             if count >= (2 if self._enable_pbn else 1):
        #                 m.eval()
        #                 # shutdown update in frozen mode
        #                 m.weight.requires_grad = False
        #                 m.bias.requires_grad = False


    def _prepare_base_model(self, base_model, config={}):
        print('=> base model: {}'.format(base_model))

        if base_model.startswith('resnet'):
            BaseResnet = IOD_ResNet(50,self.K)

            # XXXX = torchvision.models
            # self.base_model0 = getattr(BaseResnet)(True)
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model = IOD_ResNet(50,self.K)

            # url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            # model_weights = model_zoo.load_url(url)
            # self.base_model.load_state_dict(model_weights, strict=False)
            # state_dict = self.base_model.state_dict()
            # self.check_state_dict(state_dict, model_weights)
            #
            #
            # if self.is_shift:
            #     print('Adding temporal shift...')
            #     from ops.temporal_shift import make_temporal_shift
            #     make_temporal_shift(self.base_model, self.num_segments,
            #                         n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool, two_path=True)

            if self.tin:
                print('Adding temporal deformable conv...')
                from network.twod_models.temporal_interlace import make_temporal_interlace
                make_temporal_interlace(self.base_model, self.num_segments, shift_div=self.shift_div)


            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)


    def forward(self, input, no_reshape=False):
        batch_size, c, t, h, w = input.shape
        input = input.view(batch_size * self.num_segments, c ,t // self.num_segments, h, w)
        input =input.squeeze(2)
        
        base_out = self.base_model(input)#B*t,C,H,W
        bt, c, h, w = base_out.shape
        #x_output = base_out.split(bt//self.num_segments,dim = 0)
        base_out = base_out.view(batch_size,bt//batch_size,c,h,w)
        base_out = base_out.split(1,dim = 1)
        x_output = [base_out[i].squeeze(1) for i in range(len(base_out))]

        return x_output




