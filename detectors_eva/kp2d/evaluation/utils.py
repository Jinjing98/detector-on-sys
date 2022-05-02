
import numpy as np
import cv2

def draw_keypoints(img_l, vis_xyd, color=(255, 0, 0), idx=0):
    """Draw keypoints on an image"""
    # vis_xyd = top_uvz.permute(0, 2, 1)[idx].detach().cpu().clone().numpy()
    vis = img_l.copy()
    cnt = 0
    for pt in vis_xyd[:,:2].astype(np.int32):
        x, y = int(pt[1]), int(pt[0])
        cv2.circle(vis, (x,y), 2, color, -1)
    return vis


def get_proper_kpts(data):
    keypoints = data['prob'][:, :2].T
    keypoints = keypoints[::-1].T  # Y X
    prob = data['prob'][:, 2].reshape((-1,1))
    keypoints = np.hstack((keypoints,prob))

    warped_keypoints = data['warped_prob'][:, :2].T
    warped_keypoints = warped_keypoints[::-1].T
    warped_prob = data['warped_prob'][:, 2].reshape((-1,1))
    warped_keypoints = np.hstack((warped_keypoints,warped_prob))

    return  keypoints,warped_keypoints


def select_k_best(points, k):
    # k = 50
    """ Select the k most probable points (and strip their probability).
    points has shape (num_points, 3) where the last coordinate is the probability. """
    sorted_prob = points[points[:, 2].argsort(), :2]
    start = min(k, points.shape[0])
    return sorted_prob[-start:, :]




def filter_keypoints_with_ov(points, ov_region_in_warp,shape_h_w):
    """ Keep only the points whose coordinates are inside the dimensions of shape. """
    #     mask = (points[:, 0] >= 0) & (points[:, 0] < shape_h_w[0]) &\
    #            (points[:, 1] >= 0) & (points[:, 1] < shape_h_w[1])
    # print('sdfasdfadfasdf',max(points[:,0]),max(points[:,1]))
    points_ori = points.copy()
    mask_init = np.where(((points[:, 0] >= 0) & (points[:, 1] >= 0) & (points[:, 0] < shape_h_w[0]) & (points[:, 1] < shape_h_w[1])))
    points = points[mask_init]
    mask = (ov_region_in_warp[np.floor(points[:,0]).astype(int),np.floor(points[:,1]).astype(int),:]!=0)
    mask = np.all(mask,axis=1)
    actual_mask = mask_init[0][np.where(mask)[0]]

    return points_ori[actual_mask, :],actual_mask



def get_warped_pts(of,keypoints):
    warped_keypoints_gt = np.zeros_like(keypoints)
    for i,kpt in enumerate(keypoints):
        of_i = of[int(kpt[0]),int(kpt[1])]
        # print(of_i,of_i[::-1],kpt,kpt[:2]+of_i[::-1])
        # of_i = of_i[::-1]
        warped_kpt_gt = kpt[:2]+of_i[::-1]
        warped_kpt_gt = np.hstack((warped_kpt_gt,kpt[2]))
        warped_keypoints_gt[i] = warped_kpt_gt

    return warped_keypoints_gt


def select_k_best_pts_des(points, descriptors, k):
    """ Select the k most probable points (and strip their probability).
    points has shape (num_points, 3) where the last coordinate is the probability.

    Parameters
    ----------
    points: numpy.ndarray (N,3)
        Keypoint vector, consisting of (x,y,probability).
    descriptors: numpy.ndarray (N,256)
        Keypoint descriptors.
    k: int
        Number of keypoints to select, based on probability.
    Returns
    -------

    selected_points: numpy.ndarray (k,2)
        k most probable keypoints.
    selected_descriptors: numpy.ndarray (k,256)
        Descriptors corresponding to the k most probable keypoints.
    """
    sorted_prob = points[points[:, 2].argsort(), :2]
    sorted_desc = descriptors[points[:, 2].argsort(), :]
    start = min(k, points.shape[0])
    selected_points = sorted_prob[-start:, :]
    selected_descriptors = sorted_desc[-start:, :]

    return selected_points, selected_descriptors
