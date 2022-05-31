# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
from tqdm import tqdm
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from detectors_eva.utils.descriptor_evaluation import *
from detectors_eva.utils.detector_evaluation import *
from detectors_eva.utils.img_process import to_gray_normalized, to_color_normalized


def evaluate_keypoint_net_syth_data(data_loader, keypoint_net,  top_k, use_color=True,vis_flag = False):
    """Keypoint net evaluation script.

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader.
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False

    N_1,N_2 = [], []
    localization_err, repeatability = [], []
    success_count,fail_count,avg_error = [], [], []


    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            # b_size = sample['src_norm'].shape[0]
            if use_color:
                image = to_color_normalized(sample['src_norm'].cuda())
                warped_image = to_color_normalized(sample['tgt_norm'].cuda())
            else:
                image = to_gray_normalized(sample['src_norm'].cuda())
                warped_image = to_gray_normalized(sample['tgt_norm'].cuda())
            score_1s, coord_1s, desc1s = keypoint_net(image)
            score_2s, coord_2s, desc2s = keypoint_net(warped_image)
            B, _, Hc, Wc = desc1s.shape
            for b_i in range(B):
                coord_1 = coord_1s[b_i:b_i+1]
                coord_2 = coord_2s[b_i:b_i+1]
                score_1 = score_1s[b_i:b_i+1]
                score_2 = score_2s[b_i:b_i+1]
                desc1 = desc1s[b_i:b_i+1]
                desc2 = desc2s[b_i:b_i+1]


                # Scores & Descriptors
                score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
                score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
                desc1 = desc1.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
                desc2 = desc2.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()



                #estimated GT H based on GT OF

                # print(desc1.shape,score_1.shape,coord_1.shape)#coord_1: x_list,y_list
                # print(sample['src'].shape,sample['of'].shape,sample['ov'].shape,score_1.shape,desc1.shape)
                # sys.exit()
                # Prepare data for eval!
                data = {'image': sample['src'][b_i].numpy().squeeze(),# for vis
                        'image_shape' : (image.size()[2],image.size()[3]),#H W
                        'warped_image': sample['tgt'][b_i].numpy().squeeze(), #for vis
                        # 'homography': sample['homography'].squeeze().numpy(),
                        'of': sample['of'][b_i].numpy().squeeze(),  #squeeze?
                        'ov': sample['ov'][b_i].numpy().squeeze(),  #squeeze?
                        'prob': score_1,
                        'warped_prob': score_2,
                        'desc': desc1,
                        'warped_desc': desc2}

                # Compute repeatabilty and localization error

                N1, N2, rep, loc_err = compute_repeatability_new(data, keep_k_points=top_k, distance_thresh=3,vis_flag=vis_flag)
                # print('rep,loc_err',rep,loc_err)# 256 512
                N_1.append(N1)
                N_2.append(N2)
                # print(N1-N2)

                repeatability.append(rep)
                localization_err.append(loc_err)

                #des metrics
                # compute metrics for the good matched pts generated with BF(src are the topk pts) # can evaluate the wellness of des!
                fail_cnt,avg_err,success_cnt = compute_failcnt_avgerr(data,top_k,pixel_trd=5)
                # print('fail cnt, avg_err',fail_cnt,avg_err)
                fail_count.append(fail_cnt)
                avg_error.append(avg_err)
                success_count.append(success_cnt)
                # sys.exit()


    return np.nanmean(N_1),np.nanmean(N_2),np.nanmean(repeatability), np.nanmean(localization_err), \
           np.nanmean(fail_cnt),np.nanmean(avg_err),np.nanmean(success_count)

