import torch
from collections import OrderedDict
'''
    new_model_state_dict = model.state_dict()
    state_dict = model_zoo.load_url(model_urls['resnet{}'.format(depth)],
                                    map_location='cpu', progress=True)
    state_d = inflate_from_2d_model(state_dict, new_model_state_dict,
                                    skipped_keys=['fc'])
    model.load_state_dict(state_d, strict=False)
'''

def inflate_from_2d_model(state_dict_2d, state_dict_3d, skipped_keys=None, inflated_dim=2):

    if skipped_keys is None:
        skipped_keys = []
    # add 'backbone.' for orginal keys to match the current network
    state_dict_2d_copy = state_dict_2d.copy()
    state_dict_2d = OrderedDict()
    for key, value in state_dict_2d_copy.items():
        state_dict_2d['backbone.'+ key] = value


    missed_keys = []
    new_keys = []
    for old_key in state_dict_2d.keys():
        if old_key not in state_dict_3d.keys():
            missed_keys.append(old_key)
    for new_key in state_dict_3d.keys():
        if new_key not in state_dict_2d.keys():
            new_keys.append(new_key)
    print("Missed tensors: {}".format(missed_keys))# 2D 预训练模型有 而 3D 新模型没有
    print("New tensors: {}".format(new_keys))#  3D 新模型有 而 2D 预训练模型没有
    print("Following layers will be skipped: {}".format(skipped_keys))

    state_d = OrderedDict()
    unused_layers = [k for k in state_dict_2d.keys()]
    uninitialized_layers = [k for k in state_dict_3d.keys()]
    initialized_layers = []
    for key, value in state_dict_2d.items():
        skipped = False# skip的layer不必管
        for skipped_key in skipped_keys:
            if skipped_key in key:
                skipped = True
                break
        if skipped:
            continue
        new_value = value
        # only inflated conv's weights
        if key in state_dict_3d:
            if value.ndimension() == 4 and 'weight' in key:
                value = torch.unsqueeze(value, inflated_dim)
                repeated_dim = torch.ones(state_dict_3d[key].ndimension(), dtype=torch.int)
                repeated_dim[inflated_dim] = state_dict_3d[key].size(inflated_dim)
                new_value = value.repeat(repeated_dim.tolist())
            state_d[key] = new_value
            initialized_layers.append(key)
            uninitialized_layers.remove(key)
            unused_layers.remove(key)

    print("Initialized layers: {}".format(initialized_layers))
    print("Uninitialized layers: {}".format(uninitialized_layers))
    print("Unused layers: {}".format(unused_layers))

    return state_d
