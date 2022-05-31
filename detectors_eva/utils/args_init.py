from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#https://stackoverflow.com/questions/17857965/how-to-parse-a-boolean-argument-in-a-script
def ParseBoolean (b):
    # ...
    if len(b) < 1:
        raise ValueError ('Cannot parse empty string into boolean.')
    b = b[0].lower()
    if b == 't' or b == 'y' or b == '1':
        return True
    if b == 'f' or b == 'n' or b == '0':
        return False
    raise ValueError ('Cannot parse string into boolean.')





def init_args():

    # Initial args
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # dataset dir
    parser.add_argument('--dataset_prefix', type=str, default='/home/jinjing/Projects/new_data/dominik_data/')
    parser.add_argument('--dataset_prefix_pts3d', type=str, default='/home/jinjing/Projects/new_data/dominik_data/')
    parser.add_argument('--batch_size', type=int, default=20)# auto adjust within the training script




    # param for data generation
    parser.add_argument('--start_idx', type=str, default='0000')
    parser.add_argument('--step_within_pair', type=int, default=5)# auto adjust within the training script
    parser.add_argument('--step_between_pairs', type=int, default=2)# auto adjust within the training script
    parser.add_argument('--max_num_pairs_per_scene', type=int, default=45)# depending on the choice of step given fixed 100 frames

    # result excel path
    parser.add_argument('--result_path', type=str, default='/home/jinjing/Projects/detector_sysdata/results/eval.xlsx')
    parser.add_argument('--cols_name',type=int,nargs = '*',default=['param1','param2',"top_k","N1","N2","repeat","loc_error","fail_cnt","success_cnt","avg_err"])# both backbone(for old) and dataset param

    # KP2D params
    parser.add_argument('--kp2d_model_path', type=str, default="/home/jinjing/Projects/keypoints_comparision/git_src_code/kp2d/pretrained_models/v4.ckpt")
    parser.add_argument('--topk_kp2d',type=int,nargs = '*',default=[50,100,500,1000])# both backbone(for old) and dataset param


    # Superpoint params
    parser.add_argument('--sp_model_path', type=str, default="/home/jinjing/Projects/keypoints_comparision/pretrained_models/saved_models/sp_v6")
    parser.add_argument('--topk_SP',type=int,nargs = '*',default=[50,100,500,1000])# both backbone(for old) and dataset param




    # traditional name list
    parser.add_argument('--methods_set',type=int,nargs = '*',default=['orb','AKAZE','agast_sift'])#,25e-4,125e-4])# both backbone(for old) and dataset param
    # ORB params
    parser.add_argument('--num4features_set',type=int,nargs = '*',default=[100,200,400])# both backbone(for old) and dataset param
    # similar to topk; num of features is num4features if trd is 0
    parser.add_argument('--fast_trd_set',type=int,nargs = '*',default=[0,2,5,10])# both backbone(for old) and dataset param


    # AKAZE params
    parser.add_argument('--trd_set',type=int,nargs = '*',default=[1e-4,2e-4,4e-4])#,25e-4,125e-4])# both backbone(for old) and dataset param
    parser.add_argument('--diff_type_set',type=int,nargs = '*',default=[0,1,2,3])# both backbone(for old) and dataset param

    # agast sift params
    parser.add_argument('--agast_type_set',type=int,nargs = '*',default=['5_8','OAST_9_16','7_12_d','7_12_s'])#,25e-4,125e-4])# both backbone(for old) and dataset param
    parser.add_argument('--agast_trd_set',type=int,nargs = '*',default=[5,10,15])# both backbone(for old) and dataset param,












    '''
    super point head torch model params
    '''
    # tl param

    # H W we decide follwo the resize as pcv; (it can be different but need to adapt sp opt consequently)
    parser.add_argument('--weights_path', type=str, default='/home/jinjing/Projects/detector_sysdata/detectors_eva/SuperPoint_torch/model_torch/superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) dis# Remove points this close to the border.tance (default: 4).')
    parser.add_argument('--topk_sp', type=int, default=2000)# prefilter
    #have nms on will be quite time consuming during training!
    parser.add_argument('--nms_flag',type=ParseBoolean,default=False)
    # parser.add_argument('--sample_mode', type=str, default='topk')# KDT






    return parser

