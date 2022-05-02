# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Adapted from: https://github.com/rpautrat/SuperPoint/blob/master/superpoint/evaluations/descriptor_evaluation.py

import random
from glob import glob
from os import path as osp

import cv2
import numpy as np

from detectors_eva.kp2d.evaluation.utils import filter_keypoints_with_ov,select_k_best_pts_des,get_warped_pts,get_proper_kpts


def compute_failcnt_avgerr(data,top_k,pixel_trd):
    shape = data['image_shape']
    keypoints,warped_keypoints = get_proper_kpts(data)

    of = data['of']
    ov = data['ov']
    desc = data['desc']
    warped_desc = data['warped_desc']

    # simple add; without filtering
    warped_keypoints_gt = get_warped_pts(of,keypoints)
    # shape_h_w replaced with ov region?
    warped_keypoints_gt,mask_frame1 = filter_keypoints_with_ov(warped_keypoints_gt, ov,shape)
    # warped_keypoints_gt_wo_tokk_prob_filter = warped_keypoints_gt.copy()
    keypoints = keypoints[mask_frame1]
    # keypoints_wo_tokk_prob_filter = keypoints.copy()
    desc = desc[mask_frame1]

    warped_keypoints,mask_frame2 = filter_keypoints_with_ov(warped_keypoints, ov,shape)
    warped_desc = warped_desc[mask_frame2]

    warped_keypoints_gt,desc_gt = select_k_best_pts_des(warped_keypoints_gt,desc,top_k)
    keypoints,desc = select_k_best_pts_des(keypoints,desc,top_k)
    warped_keypoints,warped_desc = select_k_best_pts_des(warped_keypoints,warped_desc,top_k)



    if desc.shape[0]==0 or warped_desc.shape[0] == 0:
        return np.nan,np.nan

    # use prob topk filtered pts to compute the gt_F

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc, warped_desc)
    matches_idx_q = np.array([m.queryIdx for m in matches])
    # m_keypoints = keypoints[matches_idx_q, :]

    matches_idx_t = np.array([m.trainIdx for m in matches])
    m_warped_keypoints = warped_keypoints[matches_idx_t, :]
    m_warped_keypoints_gt = warped_keypoints_gt[matches_idx_q,:]

    # F_e, mask = cv2.findFundamentalMat(m_keypoints[:, [1, 0]],
    #                           m_warped_keypoints[:, [1, 0]], cv2.FM_RANSAC)
    #
    # F_gt, mask = cv2.findFundamentalMat(m_keypoints[:, [1, 0]],
    #                           m_warped_keypoints_gt[:, [1, 0]], cv2.FM_RANSAC)
    err_pixel = np.linalg.norm(np.array(m_warped_keypoints-m_warped_keypoints_gt),axis=1)#.mean()
    fail_mask = np.where(err_pixel>pixel_trd)
    success_mask = np.where(err_pixel<=pixel_trd)
    fail_cnt = len(fail_mask[0])
    success_cnt = len(success_mask[0])
    avg_err = err_pixel[success_mask].mean()  # only consider the succssive ones.
    return fail_cnt,avg_err,success_cnt

