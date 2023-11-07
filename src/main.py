import torch
import argparse
import yaml
import numpy as np
import random
import os
from easydict import EasyDict
from dota import setup_dota, Dota
from metrics import evaluation
from video_swin_lstm.models import build_cls, build_model_cfg
from c3d_lstm.c3d_lstm import build_c3d_lstm
from play_demo import play_demo
from play import play
from test import test
from train import train
import pickle
import utils
import glob
from torchvision import transforms
from PIL import Image
from data_transform import pad_frames
import time

def load_results(filename):
    print('load file {}'.format(filename))
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content

def get_result_filename(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'results-{:02d}.pkl'.format(epoch))

def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def parse_configs():
    parser = argparse.ArgumentParser(description='MOVAD implementation')
    parser.add_argument('--config', default="cfgs/v1.yml", help='Configuration file.')
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'play', 'demo'], help='Training or testing or play phase.')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='The number of workers to load dataset. Default: 0')
    parser.add_argument('--seed', type=int, default=123, metavar='N', help='random seed (default: 123)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epoches (default: 100)')
    parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N', help='The epoch interval of model snapshot (default: 10)')
    parser.add_argument('--epoch', type=int, default=-1, help='The epoch to restart from (training) or to eval (testing).')
    parser.add_argument('--output', default='./output/v1', help='Directory where save the output.')
    parser.add_argument('--num_videos', type=int, default=20, metavar='N', help='Number of video to play (phase = play)')
    parser.add_argument('--model_name', type=str, default='c3d_lstm', choices=['swinb_lstm', 'swint_lstm', 'c3d_lstm'])
    parser.add_argument('--demo_dir', type=str, default='frame_test')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)
    return cfg

if __name__ == "__main__":
    cfg = parse_configs()
    set_deterministic(cfg.seed)
    traindata_loader, testdata_loader = setup_dota(Dota, cfg, num_workers=cfg.num_workers, VCL=cfg.get('VCL', None), phase=cfg.phase)
    checkpoint = None
    epoch = 0
    if cfg.phase != 'play':
        if cfg.model_name == 'swinb_lstm' or cfg.model_name == 'swint_lstm':
            # use swinb_lstm.yml for swinb_lstm and swin_lstm.yml for swint_lstm
            t_model, mod_kwargs, shape_input = build_model_cfg(cfg)
            model = build_cls(cfg, t_model(**mod_kwargs), shape_input)
            try:
                checkpoint = utils.load_checkpoint(cfg)
                if cfg.phase != 'play':
                    model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch'] + 1
            except FileNotFoundError:
                print('No checkpoint found')
                cfg._no_checkpoint = True
                if cfg.epoch != -1:
                    epoch = cfg.epoch
                utils.load_pretrained(model, cfg)
        elif cfg.model_name == 'c3d_lstm':
            model = build_c3d_lstm()
            try:
                checkpoint = utils.load_checkpoint(cfg)
                if cfg.phase != 'play':
                    model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch'] + 1
            except FileNotFoundError:
                print('No checkpoint found')
                cfg._no_checkpoint = True
                if cfg.epoch != -1:
                    epoch = cfg.epoch
                utils.load_pretrained(model, cfg)

    if cfg.phase == 'train':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
        index_loss = 0
        index_guess = 0
        if checkpoint is not None:
            index_guess = checkpoint.get('index_guess', 0)
            index_loss = checkpoint.get('index_loss', 0)
        train(cfg, model, traindata_loader, optimizer, None, epoch, index_guess, index_loss)

    elif cfg.phase == 'test':
        filename = get_result_filename(cfg, epoch)
        if not os.path.exists(filename):
            if cfg.get('_no_checkpoint', False):
                raise Exception('No checkpoint to test')
            with torch.no_grad():
                test(cfg, model, testdata_loader, epoch, filename)
        content = load_results(filename)
        print(evaluation(**content))

    elif cfg.phase == 'play':
        play(cfg, testdata_loader)

    elif cfg.phase == 'demo':
        tmp = []
        img_list = sorted(glob.glob(cfg.demo_dir + "/*.jpg"))[0:150] # two last duplicate, error from extract frames
        for i in range(len(img_list)):
            image = np.asarray(Image.open(img_list[i])).astype('float32')
            tmp.append(image)
        mean = cfg.get('data_mean', [0.218, 0.220, 0.209])
        std = cfg.get('data_std', [0.277, 0.280, 0.277])
        params = {'input_shape': cfg.input_shape, 'mean': mean, 'std': std}
        transform_dict_demo = transforms.Compose([pad_frames(cfg.input_shape), transforms.Lambda(lambda x: torch.tensor(x)),
                                                  transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                                                  transforms.Lambda(lambda x: x / 255.0), transforms.Normalize(params['mean'], params['std'])])
        video_data = transform_dict_demo(np.array(tmp)).unsqueeze(0).cuda()
        video_data = torch.swapaxes(video_data, 1, 2)
        fb = cfg.NF
        t_shape = (video_data.shape[0], video_data.shape[2] - fb)
        outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)
        outputs_all = []
        toas_all = []
        teas_all = []
        # idxs_all = []
        frames_counter = []
        tic = time.time()
        with torch.no_grad():
            for i in range(fb, video_data.shape[2]):
                x = video_data[:, :, i - fb:i]
                output, rnn_state = model(x, rnn_state=None)
                output = output.softmax(dim=1)
                outputs[:, i - fb] = output[:, 1].clone()
        toc = time.time()
        total_time = (toc - tic) / video_data.shape[2]
        print('Processing time:', total_time)
        print('FPS:', 1 / total_time)
        outputs_all.append(outputs.view(-1).tolist())
        toas_all.append([0])
        teas_all.append([0])
        # idxs_all.append([26])
        frames_counter.append(video_data.shape[2])
        toas_all = np.array(toas_all).reshape(-1)
        teas_all = np.array(teas_all).reshape(-1)
        # idxs_all = np.array(idxs_all).reshape(-1)
        frames_counter = np.array(frames_counter).reshape(-1)
        filename = get_result_filename(cfg, epoch)
        with open(filename, 'wb') as f:
            # pickle.dump({'outputs': outputs_all, 'toas': toas_all, 'teas': teas_all, 'idxs': idxs_all, 'frames_counter': frames_counter}, f)
            pickle.dump({'outputs': outputs_all, 'toas': toas_all, 'teas': teas_all, 'frames_counter': frames_counter}, f)
        play_demo(cfg)