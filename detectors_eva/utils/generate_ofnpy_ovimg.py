from detectors_eva.utils.handy_sim2real import compute_OF,warp2otherview
from detectors_eva.utils.args_init import init_args
import os.path

args, unknown = init_args().parse_known_args()
dataset_prefix = args.dataset_prefix
dataset_prefix_pts3d = dataset_prefix+'/3Dcoordinates_sequences/sequences'

start_idx_str = args.start_idx
step = args.step_within_pair
successive = args.step_between_pairs
num_pair_per_scene = args.max_num_pairs_per_scene

def main():
    cnt = 0
    # sub_dir_name = os.listdir(dataset_prefix)
    scenes = os.listdir(dataset_prefix_pts3d)
    for scene_i in scenes:
        for idx in range(num_pair_per_scene):
            # data_id_src = '0000' #extracted from of_path in the future
            data_id_src = str(int(start_idx_str)+successive*idx)
            data_id_src = data_id_src.zfill(4)
            data_id_tgt = str(int(data_id_src)+step)
            if int(data_id_tgt) > 100:
                break
            data_id_tgt = data_id_tgt.zfill(4)

            # read paths
            pts3d_path1 = os.path.join(dataset_prefix,'3Dcoordinates_sequences/sequences',scene_i,'3Dcoordinates',f'coords{data_id_src}.exr')
            pts3d_path2 = os.path.join(dataset_prefix,'3Dcoordinates_sequences/sequences',scene_i,'3Dcoordinates',f'coords{data_id_tgt}.exr')
            pose_path1 = os.path.join(dataset_prefix,'cam_poses_sequences/sequences',scene_i,'cam_poses',f'pose{data_id_src}.exr')
            pose_path2 = os.path.join(dataset_prefix,'cam_poses_sequences/sequences',scene_i,'cam_poses',f'pose{data_id_tgt}.exr')
            img_path1 = os.path.join(dataset_prefix,'translation_sequences/sequences',scene_i,'translation',f'translation{data_id_src}.png')
            img_path2 = os.path.join(dataset_prefix,'translation_sequences/sequences',scene_i,'translation',f'translation{data_id_tgt}.png')

            # write paths
            opt_gt_OF_path = os.path.join(dataset_prefix,'auto_OF_sequences/sequences',scene_i,'optic_flow',f'of{data_id_src}.npy')
            opt_warped_path = os.path.join(dataset_prefix,'auto_warp_sequences/sequences',scene_i,'warped_img',f'warped{data_id_src}.png')
            compute_OF(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_gt_OF_path)
            warp2otherview(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_warped_path)
            cnt += 1
    print('expected: ', len(scenes)*num_pair_per_scene,' generated: ', cnt, ' pairs of data!')

if __name__ == '__main__':
    main()  # this function is for man made version images
