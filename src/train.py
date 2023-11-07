import os
import torch
from tqdm import tqdm
import numpy as np

def gt_cls_target(curtime_batch, toa_batch, tea_batch):
    return ((toa_batch >= 0) & (curtime_batch >= toa_batch) & ((curtime_batch < tea_batch) | (toa_batch == tea_batch)))

def train(cfg, model, traindata_loader, optimizer, lr_scheduler, begin_epoch, index_guess, index_loss):
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(eval(cfg.class_weights)).to(cfg.device), label_smoothing=cfg.get('smoothing', 0.0))
    index = index_guess
    index_l = index_loss
    fb = cfg.NF
    model.train(True)
    for e in range(begin_epoch, cfg.epochs):
        for j, (video_data, data_info) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc='Epoch: %d / %d' % (e + 1, cfg.epochs)):
            video_data = video_data.to(cfg.device, non_blocking=True)
            data_info = data_info.to(cfg.device, non_blocking=True)
            video_data = torch.swapaxes(video_data, 1, 2) # [B, F, C, W, H] -> [B, C, F, W, H]
            t_shape = (video_data.shape[0], video_data.shape[2] - fb)
            targets = torch.full(t_shape, -100).to(video_data.device)
            outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)

            video_len_orig = data_info[:, 0] # 8: VCL
            toa_batch = data_info[:, 2] # time anomaly start
            tea_batch = data_info[:, 3] # time anomaly end
            v_len = video_data.shape[2] # 8: VCL

            rnn_state = None
            losses = []
            for i in range(fb, v_len): # [1, 8]
                target = gt_cls_target(i, toa_batch, tea_batch).long()
                x = video_data[:, :, i - fb:i]
                output, rnn_state = model(x, rnn_state)
                # filter frame fillers
                flt = i >= video_len_orig
                target[flt] = -100
                output[flt] = -100
                output = output.softmax(dim=1)
                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                index_l += 1
                losses.append(loss.item())
                targets[:, i - fb] = target.clone()
                out = output.max(1)[1]
                out[target == -100] = -100
                outputs[:, i - fb] = out
            # filter not selected frames
            outputs = outputs[outputs != -100]
            targets = targets[targets != -100]
            index += 1
        print('Loss:', np.mean(losses))
        if lr_scheduler is not None:
            lr_scheduler.step()
        if (e + 1) % cfg.snapshot_interval == 0:
            lr_scheduler_state_dict = None
            if lr_scheduler is not None:
                lr_scheduler_state_dict = lr_scheduler.state_dict()
            dir_chk = os.path.join(cfg.output, 'checkpoints')
            os.makedirs(dir_chk, exist_ok=True)
            path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(e + 1))
            torch.save({'epoch': e, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                       'lr_scheduler_state_dict': lr_scheduler_state_dict, 'index_guess': index, 'index_loss': index_l}, path)