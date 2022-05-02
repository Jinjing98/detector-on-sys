from detectors_eva.utils.handy_sim2real import compute_OF,warp2otherview
import os.path
from pathlib import Path

import numpy as np

pts3d_path1 = '/home/jinjing/Projects/new_data/dominik_data/3Dcoordinates_sequences/sequences/scene_1/3Dcoordinates/coords0000.exr'
pts3d_path2 = '/home/jinjing/Projects/new_data/dominik_data/3Dcoordinates_sequences/sequences/scene_1/3Dcoordinates/coords0005.exr'
pose_path1 = '/home/jinjing/Projects/new_data/dominik_data/cam_poses_sequences/sequences/scene_1/cam_poses/pose0000.exr'
pose_path2 = '/home/jinjing/Projects/new_data/dominik_data/cam_poses_sequences/sequences/scene_1/cam_poses/pose0005.exr'
img_path1 = '/home/jinjing/Projects/new_data/dominik_data/translation_sequences/sequences/scene_1/translation/translation0000.png'
img_path2 = '/home/jinjing/Projects/new_data/dominik_data/translation_sequences/sequences/scene_1/translation/translation0005.png'
# named it with the first frame since we utilize its pts3d to compute the GT flow
opt_gt_OF_path = '/home/jinjing/Projects/new_data/dominik_data/auto_OF_sequences/sequences/scene_1/optic_flow/of0000.npy'
opt_warped_path = '/home/jinjing/Projects/new_data/dominik_data/auto_warp_sequences/sequences/scene_1/warped_img/warped0000.png'




prefix = '/home/jinjing/Projects/new_data/dominik_data/'
prefix_pts3d = '/home/jinjing/Projects/new_data/dominik_data/3Dcoordinates_sequences/sequences'
data_type = os.listdir(prefix)
scenes = os.listdir(prefix_pts3d)
# print(data_type,scenes)
step = 5
successive = 10
start_idx_str = '0000'
num_pair_per_scene = 3

for scene_i in scenes:
    for idx in range(num_pair_per_scene):

        # data_id_src = '0000' #extracted from of_path in the future
        data_id_src = str(int(start_idx_str)+successive*idx)
        data_id_src = data_id_src.zfill(4)
        data_id_tgt = str(int(data_id_src)+step)
        data_id_tgt = data_id_tgt.zfill(4)

        # print(data_id_src,data_id_tgt)


        pts3d_path1 = os.path.join(prefix,'3Dcoordinates_sequences/sequences',scene_i,'3Dcoordinates',f'coords{data_id_src}.exr')
        pts3d_path2 = os.path.join(prefix,'3Dcoordinates_sequences/sequences',scene_i,'3Dcoordinates',f'coords{data_id_tgt}.exr')
        pose_path1 = os.path.join(prefix,'cam_poses_sequences/sequences',scene_i,'cam_poses',f'pose{data_id_src}.exr')
        pose_path2 = os.path.join(prefix,'cam_poses_sequences/sequences',scene_i,'cam_poses',f'pose{data_id_tgt}.exr')
        img_path1 = os.path.join(prefix,'translation_sequences/sequences',scene_i,'translation',f'translation{data_id_src}.png')
        img_path2 = os.path.join(prefix,'translation_sequences/sequences',scene_i,'translation',f'translation{data_id_tgt}.png')
        opt_gt_OF_path = os.path.join(prefix,'auto_OF_sequences/sequences',scene_i,'optic_flow',f'of{data_id_src}.npy')
        opt_warped_path = os.path.join(prefix,'auto_warp_sequences/sequences',scene_i,'warped_img',f'warped{data_id_src}.png')


        compute_OF(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_gt_OF_path)
        warp2otherview(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_warped_path)

print('generated', len(scenes)*num_pair_per_scene, ' pairs of data!')
