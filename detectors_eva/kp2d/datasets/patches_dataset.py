# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
import os.path
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from itertools import groupby
from numpy import loadtxt
from collections import defaultdict

import sys
class syth_dataset(Dataset):
    """
    HPatches dataset class.
    Note: output_shape = (output_width, output_height)
    Note: this returns Pytorch tensors, resized to output_shape (if specified)
    Note: the homography will be adjusted according to output_shape.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    use_color : bool
        Return color images or convert to grayscale.
    data_transform : Function
        Transformations applied to the sample
    output_shape: tuple
        If specified, the images and homographies will be resized to the desired shape.
    type: str
        Dataset subset to return from ['i', 'v', 'all']:
        i - illumination sequences
        v - viewpoint sequences
        all - all sequences
    """
    def __init__(self, root_dir, use_color=True):

        super().__init__()
        self.root_dir = root_dir
        self.use_color = use_color
        paths = defaultdict(list)



        prefix = root_dir
        prefix_of = os.path.join(prefix,'auto_OF_sequences/sequences')#'/home/jinjing/Projects/new_data/dominik_data/auto_OF_sequences/sequences'
        scenes = os.listdir(prefix_of)
        step = 5
        successive = 10
        start_idx_str = '0000'
        num_pair_per_scene = 3
        for scene_i in scenes:
            for idx in range(num_pair_per_scene):

                data_id_src = str(int(start_idx_str)+successive*idx)
                data_id_src = data_id_src.zfill(4)
                data_id_tgt = str(int(data_id_src)+step)
                data_id_tgt = data_id_tgt.zfill(4)


                pts3d_path1 = os.path.join(prefix,'3Dcoordinates_sequences/sequences',scene_i,'3Dcoordinates',f'coords{data_id_src}.exr')
                pts3d_path2 = os.path.join(prefix,'3Dcoordinates_sequences/sequences',scene_i,'3Dcoordinates',f'coords{data_id_tgt}.exr')
                pose_path1 = os.path.join(prefix,'cam_poses_sequences/sequences',scene_i,'cam_poses',f'pose{data_id_src}.exr')
                pose_path2 = os.path.join(prefix,'cam_poses_sequences/sequences',scene_i,'cam_poses',f'pose{data_id_tgt}.exr')
                img_path1 = os.path.join(prefix,'translation_sequences/sequences',scene_i,'translation',f'translation{data_id_src}.png')
                img_path2 = os.path.join(prefix,'translation_sequences/sequences',scene_i,'translation',f'translation{data_id_tgt}.png')
                opt_gt_OF_path = os.path.join(prefix,'auto_OF_sequences/sequences',scene_i,'optic_flow',f'of{data_id_src}.npy')
                opt_warped_path = os.path.join(prefix,'auto_warp_sequences/sequences',scene_i,'warped_img',f'warped{data_id_src}.png')

                paths['of'].append(str(opt_gt_OF_path))
                paths['ov'].append(str(opt_warped_path))

                paths['src_img'].append(img_path1)
                paths['src_pose'].append(pose_path1)
                paths['src_pts3d'].append(pts3d_path1)

                paths['tgt_img'].append(img_path2)
                paths['tgt_pose'].append(pose_path2)
                paths['tgt_pts3d'].append(pts3d_path2)


        self.files = paths

    def __len__(self):
        return len(self.files['of'])

    def __getitem__(self, idx):
        try:
            def _read_image(path):
                img = cv2.imread(str(path))
                if self.use_color:
                    return img
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return gray
            src_img = _read_image(self.files['src_img'][idx])
            tgt_img = _read_image(self.files['tgt_img'][idx])
            # of_img = _read_image(self.files['of'][idx])
            of = np.load(self.files['of'][idx])
            ov_img = _read_image(self.files['ov'][idx])
# #128 256 3
            transform = transforms.ToTensor()
            sample = {'src': src_img, 'tgt': tgt_img, 'of': of, 'ov' : ov_img}

            # norm (0,256) to (0,1) for model prediction
            for key in ['src_norm','tgt_norm']:
                sample[key] = transform(sample[key[:3]]).type('torch.FloatTensor')
            return sample
        except Exception as e:
            print(e)

