import torch
from collections import OrderedDict
import numpy as np
import os

def get_last_epoch(filenames):
    epochs = [int(name.split('-')[1].split('.')[0]) for name in filenames]
    return filenames[np.array(epochs).argsort()[-1]]

def load_checkpoint(cfg):
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    if cfg.epoch != -1:
        path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(cfg.epoch)) # load current epoch for checkpoint
    else:
        try:
            fnames = os.listdir(dir_chk)
            path = get_last_epoch(fnames)
            path = os.path.join(dir_chk, path)
        except IndexError:
            raise FileNotFoundError()
    if not os.path.exists(path):
        raise FileNotFoundError()
    print('load file {}'.format(path))
    return torch.load(path, map_location=cfg.device)

# https://github.com/haofanwang/video-swin-transformer-pytorch
def load_pretrained(model, cfg):
    pretrained = cfg.get('pretrained', None)
    if pretrained is not None:
        print('loading pretrained {}'.format(pretrained))
        t_type = cfg.get('transformer_model', 'SwinTransformer3D')
        if t_type == 'SwinTransformer3D':
            if cfg.get('pretrained2d', False):
                print('load 2D -> 3D pretrained')
                model.model.init_weights(pretrained=pretrained)
            else:
                print('load 3D pretrained')
                checkpoint = torch.load(pretrained)
                # remove backbone key
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if 'backbone' in k:
                        name = k[9:]
                        new_state_dict[name] = v
                strict = cfg.get('pretrained_strict', True)
                # check if you want only relative_position_index
                if cfg.get('pretrained_only_relative_position_index', False):
                    print('load only relative_position_index keys')
                    strict = False
                    rel_keys = [key for key in new_state_dict.keys() if key.endswith('relative_position_bias_table')]
                    new_state_dict = {k: v for k, v in new_state_dict.items() if k in rel_keys}
                # load swin pretrained
                model.model.load_state_dict(new_state_dict, strict=strict)
        else:
            print('load 2D -> 3D pretrained')
            model.model.init_weights(pretrained=pretrained)