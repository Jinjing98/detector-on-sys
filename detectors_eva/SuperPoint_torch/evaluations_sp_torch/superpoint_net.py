import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
#share the same torch dataset as kp2d
from torch.utils.data import DataLoader, Dataset
from detectors_eva.kp2d.datasets.patches_dataset import syth_dataset
import pandas as pd

import matplotlib.pyplot as plt
import sys
from detectors_eva.SuperPoint_torch.evaluations_sp_torch.process_vis_util import vis_img,myjet
from detectors_eva.SuperPoint_torch.evaluations_sp_torch.process_vis_util import batched_pts_desc_sp
from detectors_eva.utils.args_init import init_args
# from detect_match.utils.dtc_opt_preprocess import get_KDT_pts
# from detect_match.super_point.superpoint import SuperPointFrontend

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')



class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    #for speeding up since we dont want to train
    with torch.no_grad():
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        # Descriptor Head.
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return semi, desc


class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, topk_sp,nms_flag
               ):
    self.name = 'SuperPoint'
    self.nms_dist = nms_dist
    self.cell = 8 # Size of each output cell. Keep this fixed.
    self.topk = topk_sp
    self.nms_flag = nms_flag


    # Load the network in inference mode.
    self.net = SuperPointNet()
#     if cuda:
  # Train on GPU, deploy on GPU.
    self.net.load_state_dict(torch.load(weights_path))
    self.net = self.net.cuda()
#     else:
#       # Train on GPU, deploy on CPU.
#       self.net.load_state_dict(torch.load(weights_path,
#                                map_location=lambda storage, loc: storage))
    self.net.eval()


  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.

    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run2(self, img,pts2d_gt = None):
    """ Process a numpy image to extract points and descriptors.

    topk mode rather prob_threshold;
    nms remains.

    Input
      img - b*HxW numpy float32 input image in range [0,1].
    Output
      corners - b*3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - b*256xN numpy array of corresponding unit normalized descriptors.
      heatmap -b* HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 3, 'Image must be grayscale.'

    assert img.dtype == torch.float32 , 'Image must be float32.'
    b, H, W = img.shape[0], img.shape[1], img.shape[2]
    inp = img.clone()
    inp = (inp.reshape(b, 1, H, W))

    #      x: Image pytorch tensor shaped N x 1 x H x W.
    inp = torch.autograd.Variable(inp).view(b, 1, H, W)
    inp = inp.cuda()

    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc_b = outs[0], outs[1]

    #compute softmax in GPU
    dense = torch.nn.functional.softmax(semi,dim = 1)
    # Remove dustbin.
    nodust = dense[:,:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = torch.transpose(nodust,1,3)
    nodust = torch.transpose(nodust,1,2)
    heatmap = torch.reshape(nodust, [b, Hc, Wc, self.cell, self.cell])
    heatmap = torch.transpose(heatmap,2,3)
    heatmap = torch.reshape(heatmap, [b, Hc*self.cell, Wc*self.cell])





    # no confidence threshold filtering
    self.conf_thresh = -1
    b_ids, ys, xs = torch.where(heatmap >= self.conf_thresh) # Confidence threshold.
    xs = xs.reshape((b,-1))
    ys = ys.reshape((b,-1))
    if len(xs) == 0:
      return torch.zeros((b, 3, 0)), None, None
    pts_b = torch.zeros((b, 3, xs.shape[-1])) # Populate point data sized 3xN

    #
    # if self.sample_mode == 'topk':
    #     pts_final = torch.zeros((b, 3, self.topk)) # Populate point data sized 3xN
    # elif self.sample_mode == 'KDT':
    #     pts_final = torch.zeros((b, 3, self.KDT_num)) # Populate point data sized 3xN
    pts_final = torch.zeros((b, 3, self.topk)) # Populate point data sized 3xN


    for i,pts in enumerate(pts_b):
        pts[0, :] = xs[i]
        pts[1, :] = ys[i]
        pts[2, :] = heatmap[i][ys[i], xs[i]]

        # this is quite time comsuming when run on GPU!!
        if self.nms_flag:
            # print('before nms',pts.shape)
            pts, _ = self.nms_fast(pts.numpy().copy(), H, W, dist_thresh=self.nms_dist) # Apply NMS.
            pts = torch.from_numpy(pts).cuda()
            # print('after nms',pts.shape)




        # # no remove points along border.
        # bord = 0 #self.border_remove
        # toremoveW = torch.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
        # toremoveH = torch.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
        # toremove = torch.logical_or(toremoveW, toremoveH)
        # pts = pts[:, ~toremove]

        obj_num = self.topk
        inds = torch.argsort(pts[2,:])
#         print(inds,torch.flip(inds,(0,)))
        pts = pts[:,torch.flip(inds,(0,))] # Sort by confidence.



        # if the ones after nms is smaller than obj_num; populate with its own random samples
        num = pts.shape[-1]
        np.random.seed(0)
        # random shuffle is probably not necessary; since the constructed mask won't be effected by orders of pts

        if  num < obj_num:
            print(f'Ooops, populate with its own to get top{obj_num}!')
            idx = np.random.choice(num,obj_num,replace = True)
            pts = pts[:,idx]
        else:
            pts = pts[:,:obj_num]
            #shuffle
            idx = np.random.choice(obj_num,obj_num,replace = False)
            pts = pts[:,idx]


        pts_final[i] = pts

    # --- Process descriptor.
    D = coarse_desc_b.shape[1]
    if pts_b.shape[-1] == 0:
      desc = torch.zeros((b, D, 0))
    else:
        desc = torch.zeros((b, D, self.topk))
        for i in range(b):
          coarse_desc = coarse_desc_b[i:i+1]
          # Interpolate into descriptor map using 2D point locations.
          seeds_pts = pts_final[i]
          samp_pts = seeds_pts[:2, :].clone()
          # convert to value range (-1,1) for upcoming sampling
          samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
          samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
          samp_pts = samp_pts.transpose(0, 1).contiguous()
          samp_pts = samp_pts.view(1, 1, -1, 2)
          samp_pts = samp_pts.float()
          samp_pts = samp_pts.cuda()
          desc_i = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
          desc_i = desc_i.data.reshape(D, -1)
          desc_i /= torch.linalg.norm(desc_i, axis=0)[np.newaxis, :]
          desc[i] = desc_i
    # num will affect pts but not heatmap
    pts_final = pts_final.cuda()
    desc = desc.cuda()
    heatmap = heatmap.cuda()
    return pts_final, desc, heatmap




def get_sp_model(args):

    # This class runs the SuperPoint network and processes its outputs.
    sp_model = SuperPointFrontend(weights_path=args.weights_path,
                          nms_dist=args.nms_dist,
                          topk_sp = args.topk_sp,
                          nms_flag = args.nms_flag,)
    print('==> Successfully loaded pre-trained network.')

    return sp_model

args, unknown = init_args().parse_known_args()
dataset_dir = args.dataset_prefix

sp_model = get_sp_model(args)


# init data
hp_dataset = syth_dataset(root_dir=dataset_dir, use_color=True)
data_loader = DataLoader(hp_dataset,
                         batch_size=20,
                         pin_memory=False,
                         shuffle=False,
                         num_workers=8,
                         worker_init_fn=None,
                         sampler=None)




from termcolor import colored
from detectors_eva.SuperPoint_torch.evaluations_sp_torch.evaluate import evaluate_keypoint_net_syth_data_sp
    # batched_pts_desc_sp

top_ks = args.topk_kp2d
columns = args.cols_name
df = pd.DataFrame(columns= columns)
for top_k in top_ks:
    print(colored(f'Evaluating for -- top_k {top_k}','green'))
    N1, N2, rep, loc, fail_cnt,avg_err, success_cnt = evaluate_keypoint_net_syth_data_sp(
        data_loader=data_loader,
        sp_model=sp_model,
        top_k=top_k,# use confidence?
        use_color=True,
        vis_flag=False
    )

    print('N1 {0:.3f}'.format(N1))
    print('N2 {0:.3f}'.format(N2))
    print('Repeatability {0:.3f}'.format(rep))
    print('Localization Error {0:.3f}'.format(loc))
    print('fail count {:.3f}'.format(fail_cnt))
    print('success count {:.3f}'.format(success_cnt))
    print('avg err {:.3f}'.format(avg_err))
#
#     df_curr = pd.DataFrame([[None,None,top_k,N1,N2,rep,loc,fail_cnt,success_cnt,avg_err]],
#               columns=columns)
#     df = df.append(df_curr, ignore_index=True)
# write_excel(args.result_path,'sp2',df)
