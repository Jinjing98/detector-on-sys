import torch
from mathutils import Vector, Matrix, Quaternion

# based on:
# https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_camera_matrices(pose,points3D):

	### EXTRINSIC PARAMETERS ###

	pose = pose.squeeze(axis=-1)
	location, rotation = pose[:3], pose[3:]
	location = Vector(location)
	rotation = Quaternion(rotation)
	R = rotation.to_matrix().transposed()
	T = -1*R @ location
	RT = Matrix((
		R[0][:] + (T[0],),
		R[1][:] + (T[1],),
		R[2][:] + (T[2],)
	))
	R_bcam2cv = Matrix((
		(1, 0,  0),
		(0, -1, 0),
		(0, 0, -1)
	))
	# extrinsic camera matrix (RT)
	RT = R_bcam2cv @ RT


	### INTRINSIC PARAMETERS ###

	# reshape 3D coordinates
	H = points3D.size(0)
	W = points3D.size(1)
	# get first and last pixel in the image as boundary conditions
	first_px = points3D[0,0,:]
	last_px = points3D[H-1,W-1,:]
	# convert frame to homogenous camera coordinates
	first_px = project_vec(RT@homogeneous_vec(first_px))
	last_px = project_vec(RT@homogeneous_vec(last_px))

	# height/width in image/camera space + upper left corner
	x_0, y_0 = first_px[:2]
	w_c, h_c = (last_px - first_px)[:2]
	w_i, h_i = W-1, H-1
	# scaling parameters
	a_u, a_v = w_i/w_c, h_i/h_c
	# principal points
	u_0, v_0 = -a_u*x_0, -a_v*y_0

	# intrinsic camera matrix (C)
	C = Matrix((
		(a_u,  0,u_0),
		(  0,a_v,v_0),
		(  0,  0,  1)
	))
	return torch.Tensor(C), torch.Tensor(RT), torch.Tensor(C@RT)

def homogeneous_vec(point):
	return Vector((tuple(point[:]) + (1,)))

def project_vec(point):
	return point / point[-1]

# given a set of 3D coordinates and a projection matrix,
# returns the x and y pixel coordinates of each 3D point for the given image plane.
#
# output 'xy' of shape H x W x 2
#
# !!! note that this might be a bit unintuitive:
# height (y) is indexed before with (x) but each location contains (x,y) pair
# i.e. xy[y,x,:] = [x_projected, y_projected]
def get_pixel_locations(projection_matrix,points3D):

	# reshape 3D coordinates: (H,W,3) -> (H,W,4,1) in homogenous coordinates.
	H = points3D.size(0)
	W = points3D.size(1)
	ones = torch.ones((H,W,1))
	points3D = torch.cat((points3D,ones),dim=-1).unsqueeze(dim=-1)
	# project 3D coordinates into the image plane
	xyz = torch.matmul(projection_matrix,points3D).squeeze(dim=-1)
	xy, z = torch.split(xyz,[2,1],dim=-1)
	xy = (xy / z)
	return xy

def get_gt_optical_flow(points3D_1,projection_matrix_1,projection_matrix_2):

	# pixel-wise (x,y) locations of first view.
	# this should be the identity.
	# e.g. locs_1[10,5,:] = [5,10]
	# note that due to H x W shape of the tensor, y is indexed before x.
	locs_1 = get_pixel_locations(projection_matrix_1,points3D_1)
	# compute to which (x,y) location each 3D point has moved in the next frame
	locs_1_in_view_2 = get_pixel_locations(projection_matrix_2,points3D_1)

	# compute optical flow as pixelwise (delta-x,delta-y) pairs
	optical_flow = locs_1_in_view_2 - locs_1

	return optical_flow


# differntiable warping function
# i.e. errors which were computed on the warped image can be backpropagted to the original image (e.g. for view-consistency losses)
def warp(projection_matrix,points3D,img):

	# reshape 3D coordinates: (B,H,W,3) -> (B,H,W,4,1) in homogenous coordinates.
	B = points3D.size(0)
	H = points3D.size(1)
	W = points3D.size(2)
	ones = torch.ones((B,H,W,1)).cuda()
	points3D = torch.cat((points3D,ones),dim=-1).unsqueeze(dim=-1)
	points3D = points3D.transpose(1,2).reshape(B,H*W,4,1)
	# image shape
	batch_size = img.shape[0]
	num_channels = img.shape[1]
	num_pixels = H*W
	# flatten image to shape (batch_size,num_channels,-1)
	img = img.view(batch_size,num_channels,num_pixels)
	# project 3D coordinates to 2D pixel coordinates
	indTgt = torch.squeeze(torch.matmul(projection_matrix,points3D),dim=-1)
	indTgt, z = torch.split(indTgt,[2,1],dim=-1)
	indTgt = torch.round(indTgt/z).type(torch.cuda.LongTensor)
	# overwrite pixels out of scope
	pX, pY = torch.split(indTgt,1,dim=-1)
	out_of_scope = (pX >= W) | (pX < 0) | (pY >= H) | (pY < 0)
	indTgt = torch.where(out_of_scope,torch.cuda.LongTensor([W,H-1]),indTgt)
	# convert to 1D coordinates (scatter and gather can only deal with 1D indices)
	pX, pY = torch.split(indTgt,1,dim=-1)
	indTgt = torch.squeeze(pX + pY*W, dim=-1)
	# concatenate rgb with z value
	z = z.view(batch_size,W,H).transpose(2,1).reshape(batch_size,1,-1)
	img = torch.cat((img,z),dim=1)
	num_channels = img.shape[1]
	### map RGB values into target image ###
	indSrc = torch.arange(H*W).view(H,W).transpose(1,0).reshape(-1).cuda()
	indSrc = indSrc.repeat(batch_size,num_channels,1)
	indTgt = torch.unsqueeze(indTgt,dim=1).repeat(1,num_channels,1)
	# gather RGBD values form source image
	gathered = torch.gather(img,-1,indSrc)
	# scatter into target image
	# if multiple src values are mapped to the same target location, the mean is used
	# (based on https://github.com/rusty1s/pytorch_scatter/tree/1.4.0)
	empty_img = torch.zeros((batch_size,num_channels,num_pixels+1)).cuda()
	scattered = empty_img.scatter_add(-1,indTgt,gathered)
	count = empty_img.scatter_add(-1,indTgt,torch.ones_like(gathered))
	count = torch.where(count==0,torch.cuda.FloatTensor([1]),count)
	scattered = scattered / count
	# remove pixels out of range and reshape
	scattered = scattered[:,:,:-1]
	warped = scattered.view(batch_size,num_channels,H,W)
	warped, z = torch.split(warped,[3,1],dim=1)
	# print(warped.shape,warped[0,:,:,-1])
	return warped, z

def get_z(projection_matrix,points3D):

	B = points3D.size(0)
	H = points3D.size(1)
	W = points3D.size(2)
	ones = torch.ones((B,H,W,1)).cuda()
	points3D = torch.cat((points3D,ones),dim=-1).unsqueeze(dim=-1)
	points3D = points3D.transpose(1,2).reshape(B,H*W,4,1)

	ind = torch.squeeze(torch.matmul(projection_matrix,points3D),dim=-1)
	z = ind[:,:,-1]
	z = z.view(B,1,W,H).transpose(3,2)
	return z

def remove_occlusions(warped_into_tgt,z_src,projection_matrix_tgt,points3D_tgt,epsilon=1):

	z_tgt = get_z(projection_matrix_tgt,points3D_tgt)

	occlusion = ((z_src-z_tgt) > epsilon) & (z_src != 0)
	warped_into_tgt = torch.where(occlusion,torch.cuda.FloatTensor([0]),warped_into_tgt)
	return warped_into_tgt
