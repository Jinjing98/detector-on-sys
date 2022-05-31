import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
import matplotlib.pyplot as plt
import sys

import torch

import cv2
import numpy as np






# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


def vis_img(img):

    '''
    '''
    img_resize = img
    img_resize = img_resize.astype(np.uint8) # opencv: BGR
    # for matplotlib/PIL visulisation requires RGB
    img_resize = np.flip(img_resize,axis = -1)# convert to RGB
    resize_w = img_resize.shape[1]
    resize_h = img_resize.shape[0]
    fig, ax = plt.subplots(1,1)#
    ax.imshow(img_resize)



# directly get single gray img
def read_image_gray(impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim






# used to convert batched BGR CV data to gray manually
def bgr2gray(cv_bgr_img):
    '''
    cv_bgr_img: b * h * w * 3
    '''

#     rgb = cv_bgr_img
    rgb = cv_bgr_img[: ,:, :, [2, 1, 0]]

    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray /= 255
#     print('toda',rgb.shape,rgb[0],gray.shape,gray[0])
#     sys.exit()
    return gray



def batched_pts_desc_sp(sp_model,imgs_resize):

    #pts2ds: w h
    # gray : H w
    gray = bgr2gray(imgs_resize)
    # pts_b: w h prob
    pts_b, desc_b, heatmap_b = sp_model.run2(gray,None)
    # remove the prob dim
    pts_b = pts_b[:,:2,:]
    h,w = gray.shape[-2],gray.shape[-1]
    pts_b = torch.transpose(pts_b,1,2)/torch.Tensor([w,h]).cuda()


    return pts_b,desc_b




