# Copyright 2020 Toyota Research Institute.  All rights reserved.
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.config.set_visible_devices([], 'GPU')

import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from detectors_eva.utils.descriptor_evaluation import *
from detectors_eva.utils.detector_evaluation import *

def  evaluate_keypoint_net_syth_data_sp(sess,top_k,input_img_tensor,output_prob_nms_tensor,output_desc_tensors,generators,data_size,vis_flag = False):



    N_1,N_2 = [], []
    localization_err, repeatability = [], []
    success_count,fail_count,avg_error = [], [], []

    for data in generators:

        img1 = data['src_norm']
        img2 = data['tgt_norm']
        #extract score_1 and desc1 from src
        out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img1, 0)})
        prob_map1 = np.squeeze(out1[0])
        descriptor_map1 = np.squeeze(out1[1])

        coord_1 = np.where(prob_map1 > 0)#y,x
        prob_1 = prob_map1[coord_1].reshape((-1,1))
        desc1 = descriptor_map1[coord_1]
        coord_1 = np.array(coord_1).T
        # kpts should have order: x,y
        coord_1[:,[0, 1]] = coord_1[:,[1, 0]]
        score_1 = np.hstack((coord_1,prob_1))


        #extract score_1 and desc1 from tgt
        out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
        prob_map2 = np.squeeze(out2[0])
        descriptor_map2 = np.squeeze(out2[1])

        coord_2 = np.where(prob_map2 > 0)#y,x
        prob_2 = prob_map2[coord_2].reshape((-1,1))
        desc2 = descriptor_map2[coord_2]
        coord_2 = np.array(coord_2).T
        # kpts should have order: x,y
        coord_2[:,[0, 1]] = coord_2[:,[1, 0]]
        score_2 = np.hstack((coord_2,prob_2))

        of = np.array(data['of']).squeeze(0)
        ov = data['ov']
        image1 = data['src']
        image2 = data['tgt']


        new_data = {
            'image':image1,
            'image_shape':(image1.shape[0],image1.shape[1]),
            'warped_image':image2,
            'of':of,
            'ov':ov,
            'prob':score_1,
            'warped_prob':score_2,
            'desc':desc1,
            'warped_desc':desc2,
        }

        N1, N2, rep, loc_err = compute_repeatability_new(new_data, keep_k_points=top_k, distance_thresh=3,vis_flag=vis_flag)
        fail_cnt,avg_err,success_cnt = compute_failcnt_avgerr(new_data,top_k,pixel_trd=5)

        N_1.append(N1)
        N_2.append(N2)
        repeatability.append(rep)
        localization_err.append(loc_err)
        fail_count.append(fail_cnt)
        avg_error.append(avg_err)
        success_count.append(success_cnt)
        if len(N_1) == data_size:
            print('done with evalution on this epoch!')
            break

    return np.nanmean(N_1),np.nanmean(N_2),np.nanmean(repeatability), np.nanmean(localization_err), \
           np.nanmean(fail_cnt),np.nanmean(avg_err),np.nanmean(success_count)

