# coding=utf-8
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from model.nms_wrapper import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch
label_map = ['__background__', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks',
                   'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg',
                   'Left-shoe', 'Right-shoe']
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(net, im, label=None):
  #im = cv2.imread(imdb.image_path_at(i))
  ori_img = im.copy()
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  # scores(300,num_classes) bbox_pred(300, num_classes*4) rois(300,4)
  # 对于300个roi，每个roi 4个值 x1 y1 x2 y2
  # score是对于每个roi，属于各个类别的概率(经过softmax后)(0.1 0.2 0.1.....)
  # bbox_pred是每个roi对于每个类别的坐标偏移 即同一个roi经过不同的偏移后可以属于多个类别
  if cfg.DO_PARSING:
    _, scores, bbox_pred, rois, mask_score_map = net.test_image(blobs['data'], blobs['im_info'])
  else:
    _, scores, bbox_pred, rois = net.test_image(blobs['data'], blobs['im_info'])

  # for i in range(rois.shape[0]):
  #   box = rois[i, 1:5] / im_scales[0]
  #   box = box.astype(np.int64)
  #   proposal = ori_img[box[1]:box[3],box[0]:box[2],:]
  #   cv2.imwrite('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/rois/val/'+str(i)+'.png',proposal)
  # print(rois.shape)
  # print(rois[0:10,:])

  boxes = rois[:, 1:5] / im_scales[0]  # (300,num_classes)
  scores = np.reshape(scores, [scores.shape[0], -1])  # (300,num_classes)
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])  # (300, num_classes*4)
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    # 每个roi对于不同类别进行对应偏移
    pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
    pred_boxes = _clip_boxes(pred_boxes, im.shape)  # (300, num_classes*4)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))  # test时不再回归了  (300, num_classes*4)
  # for i in range(pred_boxes.shape[0]):
  #   for j in range(1,12):
  #     if scores[i][j] > 0:
  #       proposal = ori_img[int(pred_boxes[i][j*4+1]):int(pred_boxes[i][j*4+3]),
  #                  int(pred_boxes[i][j * 4 + 0]):int(pred_boxes[i][j*4+2]), :]
  #       cv2.imwrite('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/rois/val/' + str(i)+'_'+str(j) + '_'+str(scores[i][j])+ '.png', proposal)
  # # for i in range(rois.shape[0]):
  #   box = rois[i, 1:5] / im_scales[0]
  #   box = box.astype(np.int64)
  #   proposal = ori_img[box[1]:box[3],box[0]:box[2],:]
  #   cv2.imwrite('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/rois/val/'+str(i)+'.png',proposal)
  if cfg.DO_PARSING:
    return scores, pred_boxes, mask_score_map
  return scores, pred_boxes

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(torch.from_numpy(dets), thresh).numpy()
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes

def test_net(net, imdb, weights_filename, max_per_image=100, thresh=0.,clean_pre_result=True):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)

  #num_images = 10


  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]

  output_dir = get_output_dir(imdb, weights_filename)
  det_file = os.path.join(output_dir, 'detections.pkl')
  if not clean_pre_result:
    if os.path.isfile(det_file):
      all_boxes = pickle.load(open(det_file, 'rb'))
      print('Evaluating detections')
      imdb.evaluate_detections(all_boxes, output_dir)
    else:
      print('no previous result')
  else:
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    for i in range(num_images):
      img_dir = imdb.image_path_at(i)
      img_name = img_dir[img_dir.rfind('/')+1:]
      img_name = img_name[:img_name.rfind('.')]
      im = cv2.imread(imdb.image_path_at(i))
      label = None
      _t['im_detect'].tic()
      if cfg.DO_PARSING:
        label = cv2.imread(imdb.parsing_label_path_at(i), cv2.IMREAD_GRAYSCALE)
        # scores(300,num_classes) boxes(300, num_classes*4) mask_score_map_sigmoid(300, 20, h, w)
        scores, boxes, mask_score_map = im_detect(net, im, label)
        '''
        # print(mask_score_map_sigmoid.shape)
        mask_target_channel = scores.argmax(1)  # (300,)
        cv2.imwrite('/media/rgh/rgh-data/PycharmProjects/cvpr2018/temp/mask/'+str(i)+str(i)+str(i)+'.jpg',im)
        # print(mask_target_channel)
        flag = np.zeros(cfg.CLASS_NUMS)
        for m in range(len(mask_target_channel)):
          if mask_target_channel[m]==0 or flag[mask_target_channel[m]] == 1:
            continue
          roi_mask = mask_score_map[m,mask_target_channel[m],:,:]
          flag[mask_target_channel[m]] = 1
          #print(roi_mask)
          roi_mask = np.round(roi_mask)
          roi_mask = roi_mask * 255

          # roi_mask[roi_mask > 0.5] = 1
          # roi_mask[roi_mask < 0.5] = 0
          #print(roi_mask)
          cv2.imwrite('/media/rgh/rgh-data/PycharmProjects/cvpr2018/temp/mask/'+str(i)+str(i)+str(i)+'_'+str(m)+
                      '_'+label_map[mask_target_channel[m]]+'.png',roi_mask)
        #continue
        a = raw_input()
        '''
      else:
        scores, boxes = im_detect(net, im)
      _t['im_detect'].toc()

      _t['misc'].tic()
      out = np.zeros((320,320,1),np.uint8)
      # skip j = 0, because it's the background class
      for j in range(1, imdb.num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
          .astype(np.float32, copy=False)
        keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
        cls_dets = cls_dets[keep, :]
        all_boxes[j][i] = cls_dets
        if cfg.DO_PARSING:
          parsing_select = mask_score_map[inds,j,:,:]
          parsing_select = parsing_select[keep,:,:]
          if len(cls_dets) > 0 and cls_dets[0][4] > 0.1:
            # 只取第一个
            x1 = int(cls_dets[0][0])
            y1 = int(cls_dets[0][1])
            x2 = int(cls_dets[0][2])
            y2 = int(cls_dets[0][3])
            w = int(x2-x1+1)
            h = int(y2-y1+1)
            #print ('x1: ', x1, 'y1: ', y1, 'x2: ', x2, 'y2: ', y2, 'w: ', w, 'h: ', h)
            out_part = j*parsing_select[0,:,:]
            out_part = cv2.resize(out_part,(w,h),interpolation=cv2.INTER_NEAREST)
            index_select = out[y1:y2+1,x1:x2+1,0] == 0
            out[y1:y2+1,x1:x2+1,0][index_select] = out_part[index_select]
      if cfg.DO_PARSING:
        cv2.imwrite(output_dir + '/parsing/' + img_name + '.png', out)

      #a = raw_input()


      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1]
                      for j in range(1, imdb.num_classes)])
        if len(image_scores) > max_per_image:
          image_thresh = np.sort(image_scores)[-max_per_image]
          for j in range(1, imdb.num_classes):
            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
            all_boxes[j][i] = all_boxes[j][i][keep, :]
      _t['misc'].toc()

      print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
          .format(i + 1, num_images, _t['im_detect'].average_time(),
              _t['misc'].average_time()))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

