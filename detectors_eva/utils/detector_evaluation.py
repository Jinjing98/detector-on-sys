# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Adapted from: https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/detector_evaluation.py


import numpy as np

from detectors_eva.utils.util_metrics_helper import filter_keypoints_with_ov,select_k_best,get_warped_pts,get_proper_kpts,draw_keypoints
import cv2


def compute_repeatability_new(data, keep_k_points, distance_thresh=3,vis_flag = True):
    shape_h_w = data['image_shape']
    if data['prob'].shape[0] == 0 :
        return np.nan,np.nan,np.nan,np.nan
    if data['warped_prob'].shape[0] == 0 :
        return np.nan,np.nan,np.nan,np.nan
    keypoints,warped_keypoints = get_proper_kpts(data)
    of = data['of']
    ov_region_in_warp = data['ov']
    warped_keypoints_gt = get_warped_pts(of,keypoints)

    # filter and remain pts within the frame
    # shape_h_w replaced with ov region?
    warped_keypoints_gt,mask = filter_keypoints_with_ov(warped_keypoints_gt, ov_region_in_warp,shape_h_w)
    warped_keypoints,_ = filter_keypoints_with_ov(warped_keypoints, ov_region_in_warp,shape_h_w)


    src_img = data['image']
    tgt_img = data['warped_image']


    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    warped_keypoints_gt = select_k_best(warped_keypoints_gt, keep_k_points)
    # print("keep k keypoints: ",warped_keypoints.shape,warped_keypoints_gt.shape,keep_k_points)

    # # vis check
    if vis_flag:
        # tgt_img2 = tgt_img.copy()
        keypoints = keypoints[mask]
        keypoints = select_k_best(keypoints, keep_k_points)

        src_img = draw_keypoints(src_img, keypoints[:,:2], color=(255, 0, 0), idx=0)
        tgt_img = draw_keypoints(tgt_img, warped_keypoints, color=(255, 0, 0), idx=0)
        tgt_img2 = draw_keypoints(ov_region_in_warp, warped_keypoints_gt, color=(255, 0, 0), idx=0)
        cv2.imshow('only pts is ov visiable: ori_src; ori_tat; tgt_with_src_of',np.concatenate((src_img,tgt_img,tgt_img2)))
        cv2.waitKey(0)



    # Compute the repeatability
    N1 = warped_keypoints_gt.shape[0]
    N2 = warped_keypoints.shape[0]
    # print(N1,N2)
    warped_keypoints_gt = np.expand_dims(warped_keypoints_gt, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)




    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(warped_keypoints_gt - warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    le1 = 0
    le2 = 0
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        correct1 = (min1 <= distance_thresh)
        count1 = np.sum(correct1)
        le1 = min1[correct1].sum()
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        correct2 = (min2 <= distance_thresh)
        count2 = np.sum(correct2)
        le2 = min2[correct2].sum()
    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2) if (count1+count2) else np.nan
        # print('cnt1 cnt2 le1 le2',count1,count2)
        loc_err = (le1 + le2) / (count1 + count2) if (count1+count2) else np.nan
    else:
        repeatability = np.nan#-1
        loc_err = np.nan#-1

    return count1, count2, repeatability, loc_err
