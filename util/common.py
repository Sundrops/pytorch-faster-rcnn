import cv2
import os
import numpy as np
import random
import shutil

# bgr
colormap = np.array([[0,0,0],
[0,0,128],
[0,0,254],
[0,85,0],
[51,0,169],
[0,85,254],
[85,0,0],
[220,119,0],
[0,85,85],
[85,85,0],
[0,51,85],
[128,86,52],
[0,128,0],
[254,0,0],
[220,169,51],
[254,254,0],
[169,254,85],
[85,254,169],
[0,254,254],
[0,169,254]])
def color(label):
    out = np.zeros((label.shape[0], label.shape[1],3))
    for i in range(0,len(colormap)):
        #print i
        mask = label == i
        out_mask = out[mask]
        out[mask, :] =  colormap[i,:]
        # for j in range(len(out_mask)):
        #     out[mask,:] = colormap[i,:]
    return out

# label = np.array([
#     [0,1,0,],
#     [2,3,4]
# ])
# label = np.ones((302,320))
# print label.shape
# print color(label)
def light(im1_name, im2_name):
    # im1
    im = cv2.imread(im1_name)
    im = im.astype(np.float32)
    im /= 255.
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l = im_lab[:, :, 0]
    L1_mean = np.mean(l)
    L1_std = np.std(l)

    # im2
    im = cv2.imread(im2_name)
    im = im.astype(np.float32)
    im /= 255.
    im_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l = im_lab[:, :, 0]
    L2_mean = np.mean(l)
    L2_std = np.std(l)

    if L2_std != 0:
        l = (l - L2_mean) / L2_std * L1_std + L1_mean
    l = l[:, :, np.newaxis]
    im_lab = np.concatenate((l, im_lab[:, :, 1:]), axis=2)
    im = cv2.cvtColor(im_lab, cv2.COLOR_LAB2BGR)
    im *= 255.
    return im
