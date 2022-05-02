import os.path
import torch
from PIL import Image
import cv2
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import numpy as np
from detectors_eva.utils import sim2real_util as util


pts3d_path1 = '/home/jinjing/Projects/new_data/dominik_data/3Dcoordinates_sequences/sequences/scene_1/3Dcoordinates/coords0000.exr'
pts3d_path2 = '/home/jinjing/Projects/new_data/dominik_data/3Dcoordinates_sequences/sequences/scene_1/3Dcoordinates/coords0005.exr'
pose_path1 = '/home/jinjing/Projects/new_data/dominik_data/cam_poses_sequences/sequences/scene_1/cam_poses/pose0000.exr'
pose_path2 = '/home/jinjing/Projects/new_data/dominik_data/cam_poses_sequences/sequences/scene_1/cam_poses/pose0005.exr'
img_path1 = '/home/jinjing/Projects/new_data/dominik_data/translation_sequences/sequences/scene_1/translation/translation0000.png'
img_path2 = '/home/jinjing/Projects/new_data/dominik_data/translation_sequences/sequences/scene_1/translation/translation0005.png'
# named it with the first frame since we utilize its pts3d to compute the GT flow
opt_gt_OF_path = '/home/jinjing/Projects/new_data/dominik_data/auto_OF_sequences/sequences/scene_1/optic_flow/of0000.npy'
opt_warped_path = '/home/jinjing/Projects/new_data/dominik_data/auto_warp_sequences/sequences/scene_1/warped_img/warped0000.png'


def compute_OF(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_gt_OF_path):

	#---------------#
	#-- LOAD DATA --#
	#---------------#

	# pixel-wise 3D coordinates of 2 consecutive frames
	# shape: H x W x 3
	points3D_1 = torch.Tensor(cv2.imread(pts3d_path1, cv2.IMREAD_UNCHANGED))
	points3D_2 = torch.Tensor(cv2.imread(pts3d_path2, cv2.IMREAD_UNCHANGED)) # not really required for OF computation. only to get the projection matrix.

	# camera poses
	# shape: 7 x 1 (3 location coordinates + 4 quaternion rotation coordinates (concatenated))
	cam_pose_1 = torch.Tensor(cv2.imread(pose_path1, cv2.IMREAD_UNCHANGED))
	cam_pose_2 = torch.Tensor(cv2.imread(pose_path2, cv2.IMREAD_UNCHANGED))

	# get projection matrices for both views
	_, _, projection_matrix_1 = util.get_camera_matrices(cam_pose_1,points3D_1)
	_, _, projection_matrix_2 = util.get_camera_matrices(cam_pose_2,points3D_2)


	#-------------------------------#
	#-- GROUND-TRUTH OPTICAL FLOW --#
	#-------------------------------#

	# compute optical flow
	# represented as pixel-wise (delta-x,delta-y) pairs
	# shape: H x W x 2
	optical_flow_gt = util.get_gt_optical_flow(points3D_1,projection_matrix_1,projection_matrix_2)


	#--------------------------------------------------------------#
	#-- ESTIMATED OPTICAL FLOW (GUNNAR-FARNEBACK) FOR COMPARISON --#
	#--------------------------------------------------------------#

	# load gray-scale images
	translation_1 = cv2.imread(img_path1,cv2.IMREAD_GRAYSCALE)
	translation_2 = cv2.imread(img_path2,cv2.IMREAD_GRAYSCALE)
	# estimate optical flow
	optical_flow_est = cv2.calcOpticalFlowFarneback(translation_1, translation_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)


	#----------------------------------#
	#-- VISUALIZE FLOW AS HSV IMAGES --#
	#----------------------------------#
	# from: https://www.geeksforgeeks.org/opencv-the-gunnar-farneback-optical-flow/

	# Create mask
	H, W = translation_1.shape[0], translation_1.shape[1]
	hsv_mask = np.zeros((H,W,3))
	# Make image saturation to a maximum value
	hsv_mask[..., 1] = 255

	### ground-truth flow
	optical_flow_gt = optical_flow_gt.numpy()
	#we should save this?
	os.makedirs(os.path.dirname(opt_gt_OF_path), exist_ok=True)

	np.save(opt_gt_OF_path,optical_flow_gt)
	# print('hahhah',optical_flow_gt.shape,optical_flow_gt[0,0,:])


	# convert to HSV and save
	mag, ang = cv2.cartToPolar(optical_flow_gt[..., 0], optical_flow_gt[..., 1])
	hsv_mask[..., 0] = ang * 180 / np.pi / 2
	hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	hsv_mask = np.uint8(hsv_mask)
	rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

	# os.makedirs(os.path.dirname(opt_gt_OF_path), exist_ok=True)
	# cv2.imwrite(opt_gt_OF_path, rgb_representation)
	return rgb_representation


def warp2otherview(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_warped_path):
	# images
	# shape: 3 x H x W
	translation_src = to_tensor(Image.open(img_path1))
	translation_tgt = to_tensor(Image.open(img_path2))

	# pixel-wise 3D coordinates
	# shape: H x W x 3
	points3D_src = torch.Tensor(cv2.imread(pts3d_path1, cv2.IMREAD_UNCHANGED))
	points3D_tgt = torch.Tensor(cv2.imread(pts3d_path2, cv2.IMREAD_UNCHANGED))

	# camera pose
	# shape: 7 x 1 (3 location coordinates + 4 quaternion rotation coordinates (concatenated))
	cam_pose_src = torch.Tensor(cv2.imread(pose_path1, cv2.IMREAD_UNCHANGED))
	cam_pose_tgt = torch.Tensor(cv2.imread(pose_path2, cv2.IMREAD_UNCHANGED))

	# get projection matrix
	_,_,projection_matrix_src = util.get_camera_matrices(cam_pose_src,points3D_src)
	_,_,projection_matrix_tgt = util.get_camera_matrices(cam_pose_tgt,points3D_tgt)

	# add a batch dimension to tensors and load to gpu (bc the functions were written for usage during training)
	translation_src = translation_src.unsqueeze(dim=0).cuda()
	translation_tgt = translation_tgt.unsqueeze(dim=0).cuda()
	points3D_src = points3D_src.unsqueeze(dim=0).cuda()
	points3D_tgt = points3D_tgt.unsqueeze(dim=0).cuda()
	projection_matrix_src = projection_matrix_src.unsqueeze(dim=0).cuda()
	projection_matrix_tgt = projection_matrix_tgt.unsqueeze(dim=0).cuda()



	#----------------------------------------------------------------#
	#-- WARP PIXELS (COLOR VALUES) INTO OTHER VIEWS (CAMERA POSES) --#
	#----------------------------------------------------------------#

	# warp pixels of translation_1 into view of translation_2:
	warped_into_tgt, z_warped_src = util.warp(projection_matrix_tgt,points3D_src,translation_src)
	os.makedirs(os.path.dirname(opt_warped_path), exist_ok=True)
	save_image(warped_into_tgt,opt_warped_path)

	return warped_into_tgt

# #
# # compute_OF(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_gt_OF_path)
# warp2otherview(pts3d_path1,pts3d_path2,pose_path1,pose_path2,img_path1,img_path2,opt_warped_path)
