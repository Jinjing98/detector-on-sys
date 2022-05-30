# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
from tqdm import tqdm
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from detectors_eva.utils.descriptor_evaluation import *
from detectors_eva.utils.detector_evaluation import *
from detectors_eva.utils.img_process import to_gray_normalized, to_color_normalized


def evaluate_keypoint_net_syth_data_all(method,param,data_loader,use_color=True,vis_flag = False):

    N_1,N_2 = [], []
    localization_err, repeatability = [], []
    success_count,fail_count,avg_error = [], [], []


    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if use_color:
                image = to_color_normalized(sample['src_norm'].cuda())
                warped_image = to_color_normalized(sample['tgt_norm'].cuda())
            else:
                image = to_gray_normalized(sample['src_norm'].cuda())
                warped_image = to_gray_normalized(sample['tgt_norm'].cuda())

            B, _, Hc, Wc = warped_image.shape


            for b_i in range(B):

                # method = 'agast_sift'
                if method == 'orb':
                    #
                    # num4features = 200
                    # fast_trd = 5  #similar to topk; num of features is num4features if trd is 0
                    num4features,fast_trd = param
                    top_k = 100000 # no such param for orb
                    orb = cv2.ORB_create(nfeatures=num4features,fastThreshold=fast_trd)
                    img1 = sample['src'][b_i].numpy().squeeze()
                    kp1, desc1 = orb.detectAndCompute(img1,None)  # maybe less than 500
                    #hard code the prob as the same 1
                    score_1 = np.array([[pt.pt[0],pt.pt[1],1]for pt in kp1])
                    img2 = sample['tgt'][b_i].numpy().squeeze()
                    kp2, desc2 = orb.detectAndCompute(img2,None)  # maybe less than 500
                    score_2 = np.array([[pt.pt[0],pt.pt[1],1] for pt in kp2])
                elif method == 'AKAZE':

                    # trd = 1e-4 #  [1e-4,5e-4,25e-4,125e-4]   # affaect the num of detected poitns
                    # diff_type = 0 # 0,1,2,3
                    trd, diff_type = param
                    top_k = 10000# no such param
                    akaze = cv2.AKAZE_create(threshold=trd,diffusivity=diff_type)
                    img1 = sample['src'][b_i].numpy().squeeze()
                    kp1, desc1 = akaze.detectAndCompute(img1,None)  # maybe less than 500
                    #hard code the prob as the same 1
                    score_1 = np.array([[pt.pt[0],pt.pt[1],1]for pt in kp1])
                    img2 = sample['tgt'][b_i].numpy().squeeze()
                    kp2, desc2 = akaze.detectAndCompute(img2,None)  # maybe less than 500
                    score_2 = np.array([[pt.pt[0],pt.pt[1],1] for pt in kp2])
                elif method == 'agast_sift':
                    sift = cv2.xfeatures2d.SIFT_create()
                    AGAST_TYPES = {
                                    '5_8': cv2.AgastFeatureDetector_AGAST_5_8,
                                    'OAST_9_16': cv2.AgastFeatureDetector_OAST_9_16,
                                    '7_12_d': cv2.AgastFeatureDetector_AGAST_7_12d,
                                    '7_12_s': cv2.AgastFeatureDetector_AGAST_7_12s
                    }
                    # agast_type = '5_8'
                    # trd = 5 #[5,10,15,20,25]
                    top_k = 10000# no such param

                    agast_type,trd = param
                    agast = cv2.AgastFeatureDetector_create(threshold=trd, nonmaxSuppression=True, type=AGAST_TYPES[agast_type])
                    img1 = sample['src'][b_i].numpy().squeeze()

                    kp1 = agast.detect(img1)  #  how to restrict the num of detected kps
                    kp1,desc1 = sift.compute(img1,kp1)
                    #hard code the prob as the same 1
                    score_1 = np.array([[pt.pt[0],pt.pt[1],1]for pt in kp1])
                    img2 = sample['tgt'][b_i].numpy().squeeze()
                    kp2 = agast.detect(img2)  #  how to restrict the num of detected kps
                    kp2, desc2 = sift.compute(img2,kp2)#akaze.detectAndCompute(img2,None)  # maybe less than 500
                    score_2 = np.array([[pt.pt[0],pt.pt[1],1] for pt in kp2])






                # x y
                # print(kpts1.shape,kpts2.shape,kp1[-1].pt[0],kp1[-1].pt[1],des1.shape)
                # sys.exit()



                data = {'image': sample['src'][b_i].numpy().squeeze(),# for vis
                        'image_shape' : (Hc,Wc),#H W
                        'warped_image': sample['tgt'][b_i].numpy().squeeze(), #for vis
                        # 'homography': sample['homography'].squeeze().numpy(),
                        'of': sample['of'][b_i].numpy().squeeze(),  #squeeze?
                        'ov': sample['ov'][b_i].numpy().squeeze(),  #squeeze?
                        'prob': score_1,
                        'warped_prob': score_2,
                        'desc': desc1,
                        'warped_desc': desc2}
                #2048*3 (x,y,prob)  2048*256
                # print('ssssssss',data['prob'][-1],data['desc'].shape)


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


    return np.nanmean(N_1),np.nanmean(N_2),np.nanmean(repeatability), np.nanmean(localization_err), \
           np.nanmean(fail_cnt),np.nanmean(avg_err),np.nanmean(success_count)

