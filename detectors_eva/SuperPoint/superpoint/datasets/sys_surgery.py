# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
import os.path
import cv2
import numpy as np
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
from collections import defaultdict
import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from detectors_eva.SuperPoint.superpoint.evaluations.evaluate import evaluate_keypoint_net_SP2
tf.config.set_visible_devices([], 'GPU')
#
from .base_dataset import BaseDataset

from detectors_eva.utils.tf_dataset import ratio_preserving_resize





class syth_dataset_tf(BaseDataset):

    def _init_dataset(self, root_dir = None, use_color=True):

        # super().__init__()
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


        # self.files = paths
        files = paths.copy()
        tf.data.Dataset.map_parallel = lambda self, fn: self.map(
                fn, num_parallel_calls=10) # num parallel calls

        return files

    def _get_data(self, files, split_name, **config):

        def _read_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_png(image, channels=3)

            print(image,tf.cast(image, tf.float32))
            sys.exit()
            return tf.cast(image, tf.float32)

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            # if config['preprocessing']['resize']:
            # resize = [240, 320]# False means original
            # image = ratio_preserving_resize(image,resize)
            return image

        def read_npy_file(item):
            data = np.load(item.decode('utf-8'))
            return data.astype(np.float32)



        names_tgt = tf.data.Dataset.from_tensor_slices(files['tgt_img'])
        names_src = tf.data.Dataset.from_tensor_slices(files['src_img'])
        names_ov = tf.data.Dataset.from_tensor_slices(files['ov'])
        names_of = tf.data.Dataset.from_tensor_slices(files['of'])


        tgt = names_tgt.map(_read_image)
        src = names_src.map(_read_image)
        ov = names_ov.map(_read_image)
        of = names_of.map(
                lambda item: tuple(tf.py_func(read_npy_file, [item], [tf.float32,])))


        tgt_norm = tgt.map(_preprocess)
        src_norm = src.map(_preprocess)



        data = tf.data.Dataset.zip({'tgt_norm': tgt_norm, 'tgt': tgt,
                                    'src_norm': src_norm, 'src': src,
                                    'ov':ov,
                                    'of':of,


                                    })
        data = data.map_parallel(
            lambda d: {**d, 'tgt_norm': tf.to_float(d['tgt_norm']) / 255.,
                       'src_norm': tf.to_float(d['src_norm']) / 255.,
                       })


        return data
