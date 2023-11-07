import torch
import numpy as np
import pickle
from tqdm import tqdm

def gt_cls_target(curtime_batch, toa_batch, tea_batch):
    return ((toa_batch >= 0) & (curtime_batch >= toa_batch) & ((curtime_batch < tea_batch) | (toa_batch == tea_batch)))

def test(cfg, model, testdata_loader, epoch, filename):
    targets_all = []
    outputs_all = []
    toas_all = []
    teas_all = []
    idxs_all = []
    info_all = []
    frames_counter = [] # total of frame per video

    fb = cfg.NF
    model.eval()
    for j, (video_data, data_info) in tqdm(enumerate(testdata_loader), total=len(testdata_loader), desc='Epoch: %d / %d' % (epoch, cfg.epochs)):
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)
        video_data = torch.swapaxes(video_data, 1, 2)  # [B, F, C, W, H] -> [B, C, F, W, H] -> [B=1, C=3, frame_cnt=120, W=240, H=320]
        t_shape = (video_data.shape[0], video_data.shape[2] - fb)
        targets = torch.full(t_shape, -100).to(video_data.device)
        outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)
        idx_batch = data_info[:, 1] # idx 0, 1, ... total test set
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        info_batch = data_info[:, 7:11] # [2, 1, 0, 1] --> [label, ego_involve, day or night, has object or not]
        rnn_state = None

        for i in range(fb, video_data.shape[2]):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            x = video_data[:, :, i - fb:i]
            output, rnn_state = model(x, rnn_state)
            output = output.softmax(dim=1)
            targets[:, i - fb] = target.clone()
            outputs[:, i - fb] = output[:, 1].clone()
        # collect results for each video
        targets_all.append(targets.view(-1).tolist()) # ground truth of each frame in a video
        outputs_all.append(outputs.view(-1).tolist()) # score of each frame in a video
        toas_all.append(toa_batch.tolist()) # time anomaly start
        teas_all.append(tea_batch.tolist()) # time anomaly end
        idxs_all.append(idx_batch.tolist()) # idx 0, 1,...,total number of frame
        info_all.append(info_batch.tolist())
        frames_counter.append(video_data.shape[2]) # num_frames
    # collect results for all dataset
    toas_all = np.array(toas_all).reshape(-1)
    teas_all = np.array(teas_all).reshape(-1)
    idxs_all = np.array(idxs_all).reshape(-1)
    info_all = np.array(info_all).reshape(-1, 4)
    frames_counter = np.array(frames_counter).reshape(-1)
    print('Save file {}'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump({'targets': targets_all, 'outputs': outputs_all, 'toas': toas_all, 'teas': teas_all,
                     'idxs': idxs_all, 'info': info_all, 'frames_counter': frames_counter}, f)