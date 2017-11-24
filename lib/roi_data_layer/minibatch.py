# coding=utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  blobs = {'data': im_blob}
  parsing_labels = None
  if roidb[0]['do_parsing']:
    label = cv2.imread(roidb[0]['parsing_labels'], cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, None, None, fx=im_scales[0], fy=im_scales[0],
                       interpolation=cv2.INTER_NEAREST)
    parsing_labels = np.zeros((cfg.CLASS_NUMS, label.shape[0], label.shape[1]))
    for i in range(cfg.CLASS_NUMS):
      mask = label == i
      parsing_labels[i][mask] = 1

    parsing_labels = parsing_labels[np.newaxis, ...]
    parsing_labels = parsing_labels.astype(np.float32, copy=False)
  blobs['parsing_labels'] = parsing_labels
  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"
  
  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]  # roidb[0]['gt_classes'] (num_objs, )
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd'' 
    gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  if cfg.SUB_CATEGORY:
    gt_boxes = np.ones((len(gt_inds), 6), dtype=np.float32)
    gt_boxes[:, 5] = roidb[0]['sub_categorys'][gt_inds]
  else:
    gt_boxes = np.ones((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]  # roidb[0]['boxes'] (num_objs, 4) num_objs该图实际bbox的数量
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)


  return blobs

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)


  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales