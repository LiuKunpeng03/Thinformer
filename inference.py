from itertools import product
from math import ceil
from pathlib import Path
import h5py

import warnings
import glob
import os
import pickle
import tqdm

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from torch.utils.data import Dataset
from mmcv.ops import nms

from argparse import ArgumentParser
from mmdet.utils import register_all_modules
from mmdet.utils import get_test_pipeline_cfg
from mmdet.apis import init_detector
import torch
import cv2

import time

all_time = 0

def get_multiscale_patch(sizes, steps, ratios):
    """Get multiscale patch sizes and steps.

    Args:
        sizes (list): A list of patch sizes.
        steps (list): A list of steps to slide patches.
        ratios (list): Multiscale ratios. devidie to each size and step and
            generate patches in new scales.

    Returns:
        new_sizes (list): A list of multiscale patch sizes.
        new_steps (list): A list of steps corresponding to new_sizes.
    """
    assert len(sizes) == len(steps), 'The length of `sizes` and `steps`' \
                                     'should be the same.'
    new_sizes, new_steps = [], []
    size_steps = list(zip(sizes, steps))
    for (size, step), ratio in product(size_steps, ratios):
        new_sizes.append(int(size / ratio))
        new_steps.append(int(step / ratio))
    return new_sizes, new_steps

def slide_window(width, height, sizes, steps, img_rate_thr=0.6):
    """Slide windows in images and get window position.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        sizes (list): List of window's sizes.
        steps (list): List of window's steps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        np.ndarray: Information of valid windows.
    """
    assert 1 >= img_rate_thr >= 0, 'The `in_rate_thr` should lie in 0~1'
    windows = []
    # Sliding windows.
    for size, step in zip(sizes, steps):
        size_w, size_h = size
        step_w, step_h = step

        x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
        x_start = [step_w * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size_w > width:
            x_start[-1] = width - size_w

        y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
        y_start = [step_h * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size_h > height:
            y_start[-1] = height - size_h

        start = np.array(list(product(x_start, y_start)), dtype=np.int64)
        windows.append(np.concatenate([start, start + size], axis=1))
    windows = np.concatenate(windows, axis=0)

    # Calculate the rate of image part in each window.
    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates >= img_rate_thr).any():
        img_rates[img_rates == img_rates.max()] = 1
    return windows[img_rates >= img_rate_thr]

def merge_results(results, offsets, iou_thr=0.6, device='cpu'):
    """Merge patch results via nms.

    Args:
        results (list[np.ndarray]): A list of patches results.
        offsets (np.ndarray): Positions of the left top points of patches.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """
    assert len(results) == offsets.shape[0], 'The `results` should has the ' \
                                             'same length with `offsets`.'
    merged_results = []
    for results_pre_cls in zip(*results):
        tran_dets = []
        for dets, offset in zip(results_pre_cls, offsets):
            dets[:, :2] += offset
            dets[:, 2:4] += offset
            tran_dets.append(dets)
        tran_dets = np.concatenate(tran_dets, axis=0)

        global all_time
        time_start = time.time()
        
        if tran_dets.size == 0:
            merged_results.append(tran_dets)
        else:
            tran_dets = torch.from_numpy(tran_dets)
            tran_dets = tran_dets.to(device)
            nms_dets, _ = nms(tran_dets[:, :4].contiguous(), tran_dets[:, -1].contiguous(),
                                      iou_thr)
            merged_results.append(nms_dets.cpu().numpy())
        all_time += (time.time() - time_start)
    return merged_results

def inference_detector_by_patches(model,
                                  img,
                                  sizes,
                                  steps,
                                  ratios,
                                  merge_iou_thr,
                                  bs=1):
    """inference patches with the detector.
    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.
    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray or): Either an image file or loaded image.
        sizes (list): The sizes of patches.
        steps (list): The steps between two patches.
        ratios (list): Image resizing ratios for multi-scale detecting.
        merge_iou_thr (float): IoU threshold for merging results.
        bs (int): Batch size, must greater than or equal to 1.
    Returns:
        list[np.ndarray]: Detection results.
    """

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    cfg = model.cfg

    cfg = cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    if isinstance(img[0], np.ndarray):
        # Calling this method across libraries will result
        # in module unregistered error if not prefixed with mmdet.
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    if not isinstance(img, np.ndarray):
        img = mmcv.imread(img)

    height, width = img.shape[:2]
    windows = slide_window(width, height, [(2000*3, 1200*3)], [(2000*3-600, 1200*3-600)])
    results = []
    start = 0

    time_start = time.time()
    while True:
        # prepare patch data
        patch_datas = []
        data_samples_temp = []
        if (start + bs) > len(windows):
            end = len(windows)
        else:
            end = start + bs
        for window in windows[start:end]:
            x_start, y_start, x_stop, y_stop = window
            patch = img[y_start:y_stop, x_start:x_stop]

            data = dict(img=patch, img_id=0)
            data = test_pipeline(data)
            patch_datas.append(data['inputs'])
            data_samples_temp.append(data['data_samples'])

        data['inputs'] = patch_datas
        data['data_samples'] = data_samples_temp

        with torch.no_grad():
            results_temp = model.test_step(data)
            results_temp = [[torch.cat([result.pred_instances.bboxes, result.pred_instances.scores.unsqueeze(1)],
                                  dim=1).cpu().numpy()] for result in results_temp]
            results.extend(results_temp)

        if end >= len(windows):
            break
        start += bs
    global all_time
    all_time += (time.time()-time_start)
    print(time.time()-time_start)
    results = merge_results(
        results,
        windows[:, :2],
        iou_thr=merge_iou_thr,
        device=device)
    return results

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('save_path', help='Path to save results')
    parser.add_argument(
        '--root',
        help='Root directory of the images to run inference on')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.5,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

class PANDA(Dataset):
    def __init__(self, mode="train", **kwargs):
        self.root = kwargs.get('root', '/data/liukunpeng/PANDA/image_test')
        temp = []
        self.paths = glob.glob(os.path.join(self.root, '**', '*jpg'), recursive=True)

        self.paths.sort()
        self.gt_type = kwargs['gt_type']
        if mode == "train":
            for path in self.paths:
                name = os.path.basename(path)
                tag = name.split('.')[-2].split('_')[-1]
                if tag not in ['01', '06', '11', '16', '21', '26']:
                    temp.append(path)
        else:
            for path in self.paths:
                name = os.path.basename(path)
                tag = name.split('.')[-2].split('_')[-1]
                temp.append(path)
        self.paths = temp
        self.transform = kwargs['transform']
        self.length = len(self.paths)
        self.load_raw_img = kwargs['raw']
        # self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.load_raw_img:
            img_path = self.paths[item]
            raw_path = img_path
            raw = cv2.imread(raw_path)
            name = os.path.basename(img_path)
        img, den = torch.rand(1), torch.rand(1)
        if self.load_raw_img:
            return img, den, raw, name
        return img, den

    def load_data(self, item):
        img_path = self.paths[item]
        if self.gt_type == 'adaptive_16':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_adaptive_16')
        elif self.gt_type == 'fixed_16':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_16')
        elif self.gt_type == 'adaptive_8':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_adaptive_8')
        elif self.gt_type == 'adaptive_4scale_16':
            gt_path = img_path.replace('.jpg', '.h5').replace('images_1024', 'density_map_adaptive_4scale_16')

        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(1024, 1024))

        gt_file = h5py.File(gt_path)
        den = np.asarray(gt_file['density'])

        den = cv2.resize(den, dsize=(1024, 1024))*(den.shape[0]*den.shape[1]/1024**2)
        return img, den


def main(args):
    print('save result to : ' + args.save_path)
    register_all_modules()
    all_result = []
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a huge image by patches

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset_test = PANDA(
        mode="test",
        transform=transform,
        raw=True,
        gt_type='adaptive_4scale_16',
        root=args.root)
    # print(dataset_test.__len__())
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)


    # for img in tqdm.tqdm(paths):
    for img, density, raw, name in tqdm.tqdm(dataloader_test):
        img = raw.squeeze().numpy()
        result = inference_detector_by_patches(model, img, args.patch_sizes,
                                               args.patch_steps, args.img_ratios,
                                               args.merge_iou_thr)
        all_result.append(result)
    print(all_time/len(all_result))
    with open(args.save_path, 'wb') as f:
        pickle.dump(all_result, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)