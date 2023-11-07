import os
from tqdm import tqdm
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import glob
from PIL import Image

def get_visual_directory(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'vis-{:02d}'.format(epoch))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_result_filename(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'results-{:02d}.pkl'.format(epoch))

def load_results(filename):
    print('load file {}'.format(filename))
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content

def create_curve_video(pred_scores, toa, tea, n_frames):
    # background
    fig, ax = plt.subplots(1, figsize=(30, 5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.xlabel('Frame (FPS=10)', fontsize=fontsize)
    plt.xticks(range(0, n_frames + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    curve_writer = animation.FFMpegFileWriter(fps=10, metadata=dict(title='Movie Test', artist='Matplotlib', comment='Movie support!', codec='h264'))
    with curve_writer.saving(fig, "tmp_curve_video.mp4", 100):
        xvals = np.arange(n_frames+1)
        pred_scores = pred_scores + [pred_scores[-1]]
        for t in range(1, n_frames+1):
            plt.plot(xvals[:(t+1)], pred_scores[:(t+1)], linewidth=5.0, color='r')
            plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
            if toa >= 0 and tea >= 0:
                plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                plt.axvline(x=tea, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                x = [toa, tea]
                y1 = [0, 0]
                y2 = [1, 1]
                ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            curve_writer.grab_frame()
    plt.close()
    # read frames
    cap = cv2.VideoCapture("tmp_curve_video.mp4")
    ret, frame = cap.read()
    curve_frames = []
    while ret:
        curve_frames.append(frame)
        ret, frame = cap.read()
    return curve_frames

def create_result_video(cfg, frames, curve_frames, vis_file):
    display_fps = 10
    image_size = cfg.image_shape
    video_writer = cv2.VideoWriter(vis_file, cv2.VideoWriter_fourcc(*'DIVX'), display_fps, (image_size[1], image_size[0]))
    for t, frame_vis in enumerate(frames[cfg.NF:]):
        frame_vis = frame_vis.astype(np.uint8)
        frame_vis = frame_vis[..., ::-1].copy() # rgb -> bgr
        curve_img = curve_frames[t]
        shape = curve_img.shape
        curve_height = int(shape[0] * (image_size[1] / shape[1]))
        curve_img = cv2.resize(curve_img, (image_size[1], curve_height), interpolation=cv2.INTER_AREA)
        frame_vis[image_size[0]-curve_height:image_size[0]] = cv2.addWeighted(frame_vis[image_size[0]-curve_height:image_size[0]], 0.4, curve_img, 0.6, 0)
        video_writer.write(frame_vis)

def play_demo(cfg):
    filename = get_result_filename(cfg, cfg.epoch)
    results = load_results(filename)
    v_dir = get_visual_directory(cfg, cfg.epoch)
    videos_out_dir = os.path.join(v_dir, 'videos')

    if not os.path.exists(videos_out_dir):
        os.makedirs(videos_out_dir)

    fc = results['frames_counter']
    if cfg.num_videos > -1:
        fc = fc[:cfg.num_videos]

    print('output directory {}'.format(v_dir))
    tmp = []
    img_list = sorted(glob.glob(cfg.demo_dir + "/*.jpg"))[0:150]
    for i in range(len(img_list)):
        image = np.asarray(Image.open(img_list[i])).astype('float32')
        tmp.append(image)
    frames = np.array(tmp)
    for i, counter in tqdm(enumerate(fc.tolist()), total=len(fc)):
        counter -= cfg.NF
        scores = results['outputs'][i]
        toa = max(0, results['toas'][i] - cfg.NF)
        tea = results['teas'][i] - cfg.NF
        curve_frames = create_curve_video(scores, toa, tea, counter)
        vis_file = os.path.join(videos_out_dir, 'vis_{}.avi'.format(i))
        create_result_video(cfg, frames, curve_frames, vis_file)
    os.remove('tmp_curve_video.mp4')