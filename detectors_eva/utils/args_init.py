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

    # param for data generation
    parser.add_argument('--start_idx', type=str, default='0000')
    parser.add_argument('--step_within_pair', type=int, default=5)# auto adjust within the training script
    parser.add_argument('--step_between_pairs', type=int, default=10)# auto adjust within the training script
    parser.add_argument('--max_num_pairs_per_scene', type=int, default=3)# depending on the choice of step given fixed 100 frames

    # result excel path
    parser.add_argument('--result_path', type=str, default='/home/jinjing/Projects/detector_sysdata/results/eval.xlsx')
    parser.add_argument('--cols_name',type=int,nargs = '*',default=["top_k","N1","N2","repeat","loc_error","fail_cnt","success_cnt","avg_err"])# both backbone(for old) and dataset param

    # KP2D params
    parser.add_argument('--kp2d_model_path', type=str, default="/home/jinjing/Projects/keypoints_comparision/git_src_code/kp2d/pretrained_models/v4.ckpt")
    parser.add_argument('--topk_kp2d',type=int,nargs = '*',default=[50,100,200,400])# both backbone(for old) and dataset param


    # Superpoint params
    parser.add_argument('--sp_model_path', type=str, default="/home/jinjing/Projects/keypoints_comparision/pretrained_models/saved_models/sp_v6")
    parser.add_argument('--topk_SP',type=int,nargs = '*',default=[50,100,200,400])# both backbone(for old) and dataset param



    # ORB params
    parser.add_argument('--num4features_set',type=int,nargs = '*',default=[100,200,400])# both backbone(for old) and dataset param
    # similar to topk; num of features is num4features if trd is 0
    parser.add_argument('--fast_trd_set',type=int,nargs = '*',default=[0,5,10])# both backbone(for old) and dataset param


    # AKAZE params
    parser.add_argument('--trd_set',type=int,nargs = '*',default=[1e-4,5e-4])#,25e-4,125e-4])# both backbone(for old) and dataset param
    parser.add_argument('--diff_type_set',type=int,nargs = '*',default=[0,1,2,3])# both backbone(for old) and dataset param

    # agast sift params
    parser.add_argument('--agast_type_set',type=int,nargs = '*',default=['5_8'])#,25e-4,125e-4])# both backbone(for old) and dataset param
    parser.add_argument('--agast_trd_set',type=int,nargs = '*',default=[5,10,15,20,25])# both backbone(for old) and dataset param,




    # paths to load the 2d img info and 3d pcd data
    parser.add_argument('--data_file', type=str, default='/storage/slurm/p042/result_data/MegaDepth_undistort/data_processed/megadepth_2d3d_q500ov0.35-0.75tp3-15_imsz.npy')
    parser.add_argument('--pcd_dir',type=str, default='/storage/slurm/p042/result_data/MegaDepth_undistort/data_processed/scene_points3d/')
    parser.add_argument('--img_dir', type=str,
                        default='/usr/prakt/p042/Datasets/MegaDepth_undistort/')


    # paths to save logs and cpt
    parser.add_argument('--logs_prefix', type=str, default='/storage/slurm/p042/result_logging/')
    parser.add_argument('--cpt_prefix', type=str, default='/storage/slurm/p042/result_checkpoints/')
    parser.add_argument('--subdir_name', type=str, default='test')
    #affect: backbone; tl trainer;vis check;
    parser.add_argument('--experiment_name', type=str, default=None)# 'sim' 'old' 'mod'


    '''
    dataset params
    '''
    parser.add_argument('--percent',type=float,default=1)# percentange of train and val set to shorten training time; max 1
    # auto adjust wrt the experiment name in all.py. 'old' use image
    parser.add_argument('--data_type',type=str, default=None)# 'simple' 'image'
    # the param setting for generating the GT_ labels
    parser.add_argument('--topk', type=int, default=1)# all current experiments not 5!
#     parser.add_argument('--orate_upper', type=float, default=1.0)  # can be value(>=1) or an orate upper bound.
    parser.add_argument('--resize_h_w',type=int,nargs = '*',default=[512,512])# both backbone(for old) and dataset param
    parser.add_argument('--merge_mode',type=str, default='topk')# 'topk' 'per_pair': per pair have at least min(3,topk)
    parser.add_argument('--empty_2d',type=ParseBoolean,default=False)
    parser.add_argument('--random_2d',type=ParseBoolean,default=False)
    parser.add_argument('--force_pure_inlier',type=ParseBoolean,default=True)#only inliers in the data
    parser.add_argument('--pts3d_bv_type',type=str,default='r3') # bv:r w q; 3d_pts: r3 w3 q3; indirectly affect the backbone by affecting FF axis num

    # exclusive for

    # exclusive for mod(sim) debug
    parser.add_argument('--random_3d',type=ParseBoolean,default=False)
    parser.add_argument('--shuffle_2d',type=ParseBoolean,default=False)

    # dataset prepare for OF experiments
    parser.add_argument('--toy_with_fewdata',type=ParseBoolean, default=False)
    parser.add_argument('--toy_data_split',type=str, default='val')
    parser.add_argument('--toy_data_size',type=int,default = 1)


    '''
    metric computation on full val data of OF model by traning  more rounds
    
    take care: the small traning data should be exactly the same as when you traning the OF.
    This will not edit the cpt; it just resume and train  more rounds on toy data, then logs
    compute metrics on whole val set!
    remember to also have 'toy_with_fewdata' True.
    need to manually stop the script if you want to stop more tranining.
    '''
    parser.add_argument('--OF_on_full_val_for_metric',type=ParseBoolean,default=False)

    '''
    backbone net params
    '''


    parser.add_argument('--num_latents', type=int, default=256)#6  # not used for sim
    parser.add_argument('--self_attn_num', type=int, default=6)#6  # not used for sim
    parser.add_argument('--npts_fix', type=int, default=1024)

    parser.add_argument('--opt_dim', type=int,default=3)
    # affect:the name of backbone layers to init for mod and old(not for sim); tl traning process& vis; won't affect dataset
    parser.add_argument('--label_fmt',type=str, default='bv')# 'norm' 'bv'  # not used for the bb of sim but used for tl of sim

    # shared by mod and old which use the same pcvio backbone: pcviomatching()
    parser.add_argument('--pts3d_self_attn_in_mod_flag',type=ParseBoolean,default=False)
    parser.add_argument('--pts3d_self_attn_in_mod_depth',type=int,default=12)
    parser.add_argument('--pts3d_self_attn_in_mod_head_num',type=int,default=1)

    # PE
    parser.add_argument('--queries_raw_dim', type=int, default=None)# auto adjust within the training script
    parser.add_argument('--pe_type_content',type=str,default='FF')# FF learned
    parser.add_argument('--pe_type_queries',type=str,default='FF')# FF learned
    parser.add_argument('--dim_learnPE_pcd', type=int, default=64)
    parser.add_argument('--dim_learnPE_img', type=int, default=64)
    parser.add_argument('--flag_posenc_content',type=ParseBoolean, default=True) # fourier
    parser.add_argument('--flag_posenc_queries',type=ParseBoolean, default=True) # learned


    # FF content
    # for img(old) it is 2(u,v); for mod it is 2(bv_x,bv_y);
    # although img data is 3d array; 2d_bv is 2d array since we are considering the axis num from the source of FF enc pov.
    parser.add_argument('--input_axis_content',type=int, default=2)
    parser.add_argument('--num_freq_bands_content',type=int, default=16)
    parser.add_argument('--max_freq_content',type=int, default=10) # fixed; not touch


    # FF queries
    # it can be 2/3(depends on 3d input is r or r3:auto adjust within the .py script
    parser.add_argument('--input_axis_queries',type=int, default=3)
    parser.add_argument('--num_freq_bands_queries',type=int, default=10)
    parser.add_argument('--max_freq_queries',type=int, default=10) # fixed; not touch
    parser.add_argument('--flag_pure_FF_PE_queries',type=ParseBoolean, default=False) # nerve
    parser.add_argument('--flag_pure_FF_PE_content',type=ParseBoolean, default=False)

    # exclusive for old
    parser.add_argument('--img_content_raw_dim', type=int, default=3)# RGB
#     parser.add_argument('--resize_h_w',type=int,nargs = '*',default=[512,512])# notice this is both backbone params and dataset params
    parser.add_argument('--img_FF_src',type=str, default='bv_2d') # 'bv_2d'(data dependent) 'norm_2d' 'bv_range' 'zero'
    parser.add_argument('--add_rgb_FF_src',type=ParseBoolean, default=False)
    # maybe change to full frame of detcted pts(perfect;sp;kp2d etc) option!
    # maybe: use_head_kpt_mask option is a better name
#     parser.add_argument('--perfect_dtc',type=ParseBoolean, default=False)#mask
    parser.add_argument('--mask_dtc',type=ParseBoolean, default=False)#mask
    parser.add_argument('--pad_sp_desc',type=ParseBoolean, default=False)
    parser.add_argument('--add_rand_outlier',type=ParseBoolean, default=False)
    parser.add_argument('--o_percent',type=float,default=0.0)# no upper bound
    parser.add_argument('--i_percent',type=float,default=1.0)# max 1.0
    parser.add_argument('--i_with_err',type=int, default=0)  #pixel err; must be int


    # exclusive for mod(sim)
    parser.add_argument('--mod_content_raw_dim', type=int, default=2)

    # exclusive for mod and old (pcviomatching backbone)
    parser.add_argument('--tail_3self_attn_flag',type=ParseBoolean, default=False)
    parser.add_argument('--recap_cross_attn',type=str, default=None) #'share' 'no_share'
    parser.add_argument('--self_attn_again',type=ParseBoolean, default=False)
#     parser.add_argument('--pts3d_latent',type=ParseBoolean, default=False)


    # exclusive for simple debug  : self attn param
    parser.add_argument('--self_attn_q_flag',type=ParseBoolean, default=False) # 'reg' 'cls' 'joint'
    parser.add_argument('--self_attn_c_flag',type=ParseBoolean, default=False) # 'reg' 'cls' 'joint'
    parser.add_argument('--depth_q', type=int, default=12)#6
    parser.add_argument('--depth_c', type=int, default=12)#6
    parser.add_argument('--head_num_q', type=int, default=1)#6
    parser.add_argument('--head_num_c', type=int, default=1)#6


    '''
    pl DATA modules
    '''
#     parser.add_argument('--data_type',type=str, default=None)# 'simple' 'image' # won't affect tl trainer!


    '''
    pl training params
    '''
    parser.add_argument('--version_num', type=int,default=0)
    parser.add_argument('--num_epoch', type=int)
    #overwrite log and (only last not best)cpt rather resume if there it is.
    parser.add_argument('--force_scratch', type=ParseBoolean,default = False)
#     parser.add_argument('--OF_on_full_val_for_metric',type=ParseBoolean,default=False)# this is both data + pltrain params.
#     parser.add_argument('--label_fmt',type=str, default='bv')# 'norm' 'bv'  # not used for the bb of sim but used for tl of sim



    # hyper params to train the network
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize_train',type=int, default=32)#32
    parser.add_argument('--batchsize_val',type=int, default=32)#32

    # loss_design
    parser.add_argument('--weight_of_l2', type=float, default=1)#loss_cls +100*loss_l2_P_gt
    parser.add_argument('--l2loss_pos_trd', type=float, default=0.5)
    parser.add_argument('--loss_mode', type=str, default='non_weighted')# not implemented
    parser.add_argument('--loss_type',type=str, default='reg') # 'reg' 'cls' 'joint'
#     parser.add_argument('--regloss2pixel',type=ParseBoolean, default=False)  #not implemented


    # exclusive for old
    # can be actually on when perfect_dtc is on
    # maybe set toe dtc_head type in the future for generalization
#     parser.add_argument('--use_sp_gt',type=ParseBoolean, default=False)  #not implemented
    parser.add_argument('--dtc_head_type',type=str, default='SP') # 'SP' 'SIFT' 'gt'



    # exclusive for mod

    # code clean debug
    parser.add_argument('--clean',type=ParseBoolean,default=True)

    # only for vis check(used for model not data)
    # properly load old FF pe model(the FF dim implementation in new nersions is different)
    parser.add_argument('--old_FF_for_vis_load_flag',type=ParseBoolean,default=False)




    # there are two parts of hyperparams: data and model(arch/training)
    # there are two places(tl data module; tl metric) where you can save hyper for tb visulisation.







    '''
    super point head torch model params
    '''
    # tl param

    # H W we decide follwo the resize as pcv; (it can be different but need to adapt sp opt consequently)
    parser.add_argument('--weights_path', type=str, default='/usr/prakt/p042/Projs/new_Percevier/detect-match-2d3d-cp/mid_results/superpoint_v1.pth',
      help='Path to pretrained weights file (default: superpoint_v1.pth).')
    parser.add_argument('--border_remove', type=int, default=0,
      help=' Remove points this close to the border.')
    parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) dis# Remove points this close to the border.tance (default: 4).')
    parser.add_argument('--conf_thresh', type=float, default=0,#0.015,
      help='Detector confidence threshold (default: 0.015).')# used for topk?
    parser.add_argument('--topk_sp', type=int, default=1024)
    parser.add_argument('--KDT_num', type=int, default=1024)
    parser.add_argument('--KDT_sp_prob_trd', type=float, default=0.0005)
    parser.add_argument('--KDT_nbr_trd', type=float, default=1)#pixel unit
    #have nms on will be quite time consuming during training!
    parser.add_argument('--nms_flag',type=ParseBoolean,default=False)
    parser.add_argument('--sample_mode', type=str, default='topk')# KDT

    #SIFT
    parser.add_argument('--sift_upper_bound', type=int, default=2000)






    #////////////////////////
    '''
    parapms just for dtc head
    '''
    # backbone
    parser.add_argument('--cross3d_dtc',type=ParseBoolean, default=False)
    parser.add_argument('--self_attn_again_dtc',type=ParseBoolean, default=False)
    parser.add_argument('--self_attn_again_dtc_depth', type=int, default=6)



    # tl
    parser.add_argument('--reg_src',type=str, default='inlier') # 'inlier'  'all'








    return parser

