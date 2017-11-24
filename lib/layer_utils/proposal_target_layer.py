# coding=utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps
import torch.nn.functional as F

import torch
from torch.autograd import Variable

def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes, parsing_labels=None):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois  # (2000, 5) 训练时RPN_POST_NMS_TOP_N=2000 测试时RPN_POST_NMS_TOP_N=300 (300, 5)
  all_scores = rpn_scores  # (2000, 1)

  # Include ground-truth boxes in the set of candidate rois
  # 训练时把gtbox也送到后面的fast进行分类和回归
  if cfg.TRAIN.USE_GT:
    #  设该张图片gtbox有15个  gt_boxes(15, 5) 0-3是x1 y1 x2 y2 4是所属cls
    zeros = rpn_rois.data.new(gt_boxes.shape[0], 1)  # 假zeros(15,1)个0
    all_rois = torch.cat(
      (all_rois, torch.cat((zeros, gt_boxes[:, :-1]), 1)), 0)  # (15, 1+4) -> (2000+15, 1+4) 忽略gt_boxes的cls 只考虑是不是object
    # not sure if it a wise appending, but anyway i am not using it
    all_scores = torch.cat((all_scores, zeros), 0)  # (2000, 1) -> (2000+15, 1)

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))  # 0.25*256

  # Sample rois with classification labels and bounding box regression
  # targets
  if cfg.SUB_CATEGORY:
    if cfg.DO_PARSING:
      labels, sub_labels, rois, roi_scores, bbox_targets, bbox_inside_weights, mask_unit = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes, parsing_labels)
    else:
      labels, sub_labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)
    sub_labels = sub_labels.view(-1, 1)
  else:
    if cfg.DO_PARSING:
      labels, rois, roi_scores, bbox_targets, bbox_inside_weights, mask_unit = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes, parsing_labels)
    else:
      labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)

  rois = rois.view(-1, 5)
  roi_scores = roi_scores.view(-1)
  labels = labels.view(-1, 1)
  bbox_targets = bbox_targets.view(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
  bbox_outside_weights = (bbox_inside_weights > 0).float()
  if cfg.SUB_CATEGORY:
    if cfg.DO_PARSING:
      return rois, roi_scores, labels, sub_labels, Variable(bbox_targets), Variable(bbox_inside_weights), Variable(
        bbox_outside_weights), mask_unit
    else:
      return rois, roi_scores, labels, sub_labels, Variable(bbox_targets), Variable(bbox_inside_weights), Variable(
        bbox_outside_weights)


  else:
    if cfg.DO_PARSING:
      return rois, roi_scores, labels, Variable(bbox_targets), Variable(bbox_inside_weights), Variable(
        bbox_outside_weights), mask_unit
    else:
      return rois, roi_scores, labels, Variable(bbox_targets), Variable(bbox_inside_weights), Variable(bbox_outside_weights)

#  bbox_target_data (256, 5) 类别和4个回归值
#  函数作用，产生两个（len(rois),4*21）大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh），
#  另一个对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
#  bbox_targets roi往其匹配的gt类别的回归（dx,dy,dw,dh） 其他类别的回归为0
#  bbox_inside_weights 各个回归的权重 只在匹配的gt类别处设为 1,1,1,1 ，其他位置为0
def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """
  # Inputs are tensor

  clss = bbox_target_data[:, 0]  # (256,)
  bbox_targets = clss.new(clss.numel(), 4 * num_classes).zero_()  # (256,4*num_classes)
  bbox_inside_weights = clss.new(bbox_targets.shape).zero_()  # (256,4*num_classes)
  inds = (clss > 0).nonzero().view(-1)  # 非0(背景)的索引 (n,) eg. [1,3] 对应的类别是8 10
  if cfg.SUB_CATEGORY:
    bbox_sub_targets = clss.new(clss.numel(), 4 * ((num_classes-1)*3+1)).zero_()
  if inds.numel() > 0:
    clss = clss[inds].contiguous().view(-1,1)  # (n, 1)
    dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)  # (n, 4) [ [1,1,1,1],[3,3,3,3] ]
    # 只第clss(gt类别)个位置赋值，其他类别处不管[ [4*8,4*8+1,4*8+2,4*8+3],[40,41,42,43] ]
    dim2_inds = torch.cat([4*clss, 4*clss+1, 4*clss+2, 4*clss+3], 1).long()
    # dim1_inds为行 dim2_inds为列 第1行第32列...第3行第43列 这8个数等于bbox_target_data[(1,3)][:, 1:]八个回归值
    bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
    bbox_inside_weights[dim1_inds, dim2_inds] = bbox_targets.new(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).view(-1, 4).expand_as(dim1_inds)

  return bbox_targets, bbox_inside_weights
#  这个是tf版本和上面的函数作用一样不过看着更清晰
def tf_get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    #取标签
    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    #生成一个全零矩阵，大小（len(rois),4*21）
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    #生成一个全零矩阵，大小同样为（len(rois),4*21）
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    #取出fg-roi的index，np.where返回的是一个tuple，tuple里存的是array，所以用[0]来去掉tuple外套
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        #对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh）
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        #对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""
  # Inputs are tensor

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4
  #  返回roi相对与其匹配的gt (dx,dy,dw,dh)四个回归值，shape（len（rois），4）
  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    #  是否归一化
    #  Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  # labels.unsqueeze(1) -> (128, 1)
  return torch.cat(
    [labels.unsqueeze(1), targets], 1)  # (128, 5) 类别和4个回归值

def gen_mask_parsing_labels(parsing_labels, mask_rois):
  # parsing_labels (48, 320, 320)  mask_rois (48, 5)
  # rois = rois.detach()

  x1 = mask_rois[:, 1::4]
  y1 = mask_rois[:, 2::4]
  x2 = mask_rois[:, 3::4]
  y2 = mask_rois[:, 4::4]

  height = parsing_labels.size(1)
  width = parsing_labels.size(2)
  # affine theta
  theta = Variable(mask_rois.data.new(mask_rois.size(0), 2, 3).zero_())
  theta[:, 0, 0] = (x2 - x1) / (width - 1)
  theta[:, 0, 2] = (x1 + x2 - width + 1) / (width - 1)
  theta[:, 1, 1] = (y2 - y1) / (height - 1)
  theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1)
  pre_pool_size = cfg.POOLING_SIZE * 8
  grid = F.affine_grid(theta, torch.Size((mask_rois.size(0), 1, pre_pool_size, pre_pool_size)))
  mask_parsing_labels = F.grid_sample(parsing_labels.unsqueeze(1), grid) # (48,1, 320, 320)
  mask_parsing_labels = torch.round(mask_parsing_labels)
  # mask_parsing_labels[mask_parsing_labels >=0.5] = 1
  # mask_parsing_labels[mask_parsing_labels < 0.5] = 0
  return mask_parsing_labels
def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, parsing_labels=None):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  """
  # overlaps: (rois x, gt_boxes)  (2000, 15)
  # 每个roi和每个gt box的iou
  overlaps = bbox_overlaps(
    all_rois[:, 1:5].data,
    gt_boxes[:, :4].data)
  max_overlaps, gt_assignment = overlaps.max(1)  # 对于每个roi，它与所有gtboxes中iou最大的作为它的gt
  # max_overlaps 每个roi与给它指定的gtbox之间的iou
  labels = gt_boxes[gt_assignment, [4]]  # 每个roi被指定的cls  gt_boxes(15,5)
  if cfg.SUB_CATEGORY:
    sub_labels = gt_boxes[gt_assignment, [5]]


  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = (max_overlaps >= cfg.TRAIN.FG_THRESH).nonzero().view(-1)
  #print(fg_inds)
  # Guard against the case when an image has fewer than fg_rois_per_image
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  # 0.1-0.5的被看成是背景
  bg_inds = ((max_overlaps < cfg.TRAIN.BG_THRESH_HI) + (max_overlaps >= cfg.TRAIN.BG_THRESH_LO) == 2).nonzero().view(-1)

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.numel() > 0 and bg_inds.numel() > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(fg_rois_per_image), replace=False)).long().cuda()]
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.numel() < bg_rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(bg_rois_per_image), replace=to_replace)).long().cuda()]
  elif fg_inds.numel() > 0:
    to_replace = fg_inds.numel() < rois_per_image
    fg_inds = fg_inds[torch.from_numpy(npr.choice(np.arange(0, fg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = rois_per_image
  elif bg_inds.numel() > 0:
    to_replace = bg_inds.numel() < rois_per_image
    bg_inds = bg_inds[torch.from_numpy(npr.choice(np.arange(0, bg_inds.numel()), size=int(rois_per_image), replace=to_replace)).long().cuda()]
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  if cfg.DO_PARSING:
    mask_rois = all_rois[fg_inds]#.contiguous()
    #print(mask_rois.size()) (64,5)
    mask_cls_labels = labels[fg_inds]#.contiguous()
    #print(mask_cls_labels.size())
    assert parsing_labels.size(0) == 1
    # parsing_labels (48, 320, 320) -> (48, 1, 28, 28)
    # TODO : parsing_labels = parsing_labels[0][gt_assignment[fg_inds], :, :]
    # gt_assignment只是指定的gt box序号，但是label是对应gtbox的第五个维度，也就是说第1个box的label不一定是1
    # print (gt_assignment.size(), labels.size())
    # print (type(gt_assignment), type(labels))
    parsing_labels = parsing_labels[0][labels.data.long()[fg_inds], :, :]  # batch channel h w  batch == 1
    #print(parsing_labels.size())

    mask_parsing_labels = gen_mask_parsing_labels(parsing_labels, mask_rois)
    #print(mask_parsing_labels.size())
    mask_unit = {}
    mask_unit['mask_rois'] = mask_rois
    mask_unit['mask_cls_labels'] = mask_cls_labels
    mask_unit['mask_parsing_labels'] = mask_parsing_labels
    #print(mask_unit['mask_parsing_labels'].size())

  # The indices that we're selecting (both fg and bg)
  keep_inds = torch.cat([fg_inds, bg_inds], 0)   # 2000个roi中256个被选为fg和bg的索引
  # Select sampled values from various arrays:
  labels = labels[keep_inds].contiguous()  # 被选中roi的cls label (256,)
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0  # 将背景的label固定为0  (256,)
  if cfg.SUB_CATEGORY:
    sub_labels = sub_labels[keep_inds].contiguous()
    sub_labels[int(fg_rois_per_image):] = 0
  rois = all_rois[keep_inds].contiguous()  # 只留下被选中的roi
  roi_scores = all_scores[keep_inds].contiguous()  # 只留下被选中roi的score(rpn网络预测这个roi为物体的概率)

  # if cfg.DO_PARSING:
  #   mask_unit = {}
  #   mask_unit['mask_rois'] = rois[:int(fg_rois_per_image),...]
  #   mask_unit['mask_cls_labels'] = labels[:int(fg_rois_per_image)]
  #   mask_unit['mask_parsing_labels'] = parsing_labels[0][labels[:int(fg_rois_per_image)], :, :]


  #  把被选择的roi和给它指定的gtbox的坐标和类别 送入_compute_targets
  #  roi的坐标 rois[:, 1:5].data(256, 4)
  #  匹配的gt的坐标 gt_boxes[gt_assignment[keep_inds]][:, :4].data (256, 4)
  #  匹配的类别 labels.data(256,)
  #  返回 (256, 5) 类别和4个回归值

  bbox_target_data = _compute_targets(
  rois[:, 1:5].data, gt_boxes[gt_assignment[keep_inds]][:, :4].data, labels.data)

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
  if cfg.SUB_CATEGORY:
    if cfg.DO_PARSING:
      return labels, sub_labels, rois, roi_scores, bbox_targets, bbox_inside_weights, mask_unit
    else:
      return labels, sub_labels, rois, roi_scores, bbox_targets, bbox_inside_weights
  else:
    if cfg.DO_PARSING:
      return labels, rois, roi_scores, bbox_targets, bbox_inside_weights, mask_unit
    else:
      return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
