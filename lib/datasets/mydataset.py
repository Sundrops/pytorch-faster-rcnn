# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .mydataset_eval import mydataset_eval
from model.config import cfg
import matplotlib.pyplot as plt
import cv2
class mydataset(imdb):
  def __init__(self, image_set, dataset):

    name = dataset + '_' + image_set
    imdb.__init__(self, name)
    self._image_set = image_set


    self._root_path = '/media/rgh/rgh-data/Dataset/'+dataset#.capitalize()
    self._data_path = os.path.join(self._root_path, 'image', image_set)
    self._parsing_label_path = os.path.join(self._root_path, 'label', image_set)
    # self._classes = ('__background__',  # always index 0
    #                  'face', 'hair', 'U-clothes', 'L-arm','R-arm',
    #                  'pants', 'L-leg', 'R-leg', 'dress','L-shoe',
    #                  'R-shoe')
    self._classes = ('__background__', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks',
                   'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg',
                   'Left-shoe', 'Right-shoe')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = '.jpg'
    self._parsing_label_ext = '.png'
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self.config = {'cleanup': False,
                   'use_diff': True,
                   'matlab_eval': False,
                   'rpn_file': None,
                   'min_size': 2}
    assert os.path.exists(self._root_path), \
      'dataset path does not exist: {}'.format(self._root_path)
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)
  def parsing_label_path_at(self, i):
    return self.parsing_label_path_from_index(self._image_index[i])

  def parsing_label_path_from_index(self, index):
    parsing_label_path = os.path.join(self._parsing_label_path,
                              index + self._parsing_label_ext)
    assert os.path.exists(parsing_label_path), \
      'Path does not exist: {}'.format(parsing_label_path)
    return parsing_label_path
  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path,
                              index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    image_set_file = os.path.join(self._root_path,
                                  self._image_set + '.txt')

    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    return image_index


  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    # if os.path.exists(cache_file):
    #   with open(cache_file, 'rb') as fid:
    #     try:
    #       roidb = pickle.load(fid)
    #     except:
    #       roidb = pickle.load(fid, encoding='bytes')
    #   print('{} gt roidb loaded from {}'.format(self.name, cache_file))
    #   return roidb

    # gt_roidb = [self._load_annotation(index)
    #             for index in self.image_index]
    gt_roidb = []
    error_index = []
    for index in self.image_index:
      roi = self._load_annotation(index)
      if roi is not None:
        gt_roidb.append(roi)
      else:
        error_index.append(index)
    for index in error_index:
      self.image_index.remove(index)
    print('error images number: ', len(error_index))
    print(error_index)
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_annotation(self, img_index):
    """
    Load image and bounding boxes info from txt file.
    """
    bbox_txt = open(os.path.join(self._root_path, 'bbox', self._image_set, img_index+'.txt'),'r')
    bboxs = []
    num_objs = 0
    for index, line in enumerate(bbox_txt):
      line = line.strip()
      if line != '':
        num_objs += 1
        #print(line.split(' '))
        x1, y1, x2, y2 = line.split(' ')
        # Make pixel indexes 0-based
        x1 = float(x1)# - 1 #if float(x1) > 1 else float(x1)
        y1 = float(y1)# - 1 #if float(y1) > 1 else float(y1)
        x2 = float(x2)# - 1
        y2 = float(y2)# - 1
        # 0 bk
        bboxs.append([index+1, [x1, y1, x2, y2],(x2 - x1 + 1) * (y2 - y1 + 1)])
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    for index, bbox in enumerate(bboxs):
      boxes[index, :] = bbox[1]
      gt_classes[index] = bbox[0]
      overlaps[index, bbox[0]] = 1.0
      seg_areas[index] = bbox[2]
    overlaps = scipy.sparse.csr_matrix(overlaps)
    if cfg.SUB_CATEGORY:
      sub_category_txt = open(os.path.join(self._root_path, 'sub_category_3', self._image_set, img_index + '.txt'), 'r')
      subs = []
      num_subs = 0
      for index, line in enumerate(sub_category_txt):
        line = line.strip()
        if line != '':
          num_subs += 1
          main, sub = line.split(' ')
          main = int(main)
          sub = int(sub)
          subs.append((main-1)*3+sub)
      # print(img_index, ' ', num_objs,' ', num_subs)
      if num_objs !=num_subs:
        return None
      sub_categorys = np.zeros((num_subs), dtype=np.int32)
      for index, sub in enumerate(subs):
        sub_categorys[index] = sub
      return {'boxes': boxes,
              'gt_classes': gt_classes,
              'sub_categorys': sub_categorys,
              'gt_overlaps': overlaps,
              'flipped': False,
              'seg_areas': seg_areas,
              'do_parsing': cfg.DO_PARSING}
    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas,
            'do_parsing': cfg.DO_PARSING}


  def _get_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._root_path,'detection_result', filename)
    return path

  def _write_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} results file'.format(cls))
      filename = self._get_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(self._root_path, 'bbox',self._image_set, '{:s}.txt')
    imagesetfile = os.path.join(self._root_path, self._image_set + '.txt')
    cachedir = os.path.join(self._root_path, 'annotations_cache')
    aps = []
    recs = []  ### ADDED
    precs = []  ### ADDED
    nposes = []  ### ADDED
    tps = []  ### ADDED
    fps = []  ### ADDED
    # The PASCAL VOC metric changed in 2010
    #use_07_metric = True if int(self._year) < 2010 else False
    #print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    use_07_metric = False
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_results_file_template().format(cls)
      # add npos, tp, fp
      rec, prec, ap, npos, tp, fp = mydataset_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=cfg.TEST.IOU_THRESH,
        use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      # plt.plot(rec, prec, lw=2, label=cls)
      # plt.xlabel('Recall')
      # plt.ylabel('Precision')
      # plt.grid(True)
      # plt.ylim([0.0, 1.05])
      # plt.xlim([0.0, 1.0])
      # plt.title('Precision-Recall')
      # plt.legend(loc="upper right")
      # plt.show()
      tps += [tp[-1]]  ### ADDED
      fps += [fp[-1]]  ### ADDED
      nposes += [npos]  ### ADDED
      aps += [ap]
      recs += [rec[-1]]
      precs += [prec[-1]]
      #print(('AP for {} = {:.4f}'.format(cls, ap)))
      print(('{} AP: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(cls, ap, rec[-1], prec[-1])))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    overall_recs = np.sum(tps) / float(np.sum(nposes))  ### ADDED
    overall_precs = np.sum(tps) / np.maximum(np.sum(tps) + np.sum(fps), np.finfo(np.float64).eps)  ### ADDED
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('Overall Recall = {:.4f}'.format(overall_recs))  ### ADDED
    print('Overall Precision = {:.4f}'.format(overall_precs))  ### ADDED
    print('Overall F1 = {:.4f}'.format((2 * overall_recs * overall_precs) / (overall_recs + overall_precs)))  ### ADDED
    print('~~~~~~~~')
    print('AP Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Recall Results:')
    for rec in recs:
      print(('{:.3f}'.format(rec)))
    print(('{:.3f}'.format(np.mean(overall_recs))))
    # print('~~~~~~~~')
    # print('Prec Results:')
    # for prec in precs:
    #   print(('{:.3f}'.format(prec)))
    # print(('{:.3f}'.format(np.mean(overall_precs))))
    # print('~~~~~~~~')
    # print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


  def evaluate_detections(self, all_boxes, output_dir):
    self._write_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_results_file_template().format(cls)
        os.remove(filename)


if __name__ == '__main__':
  from mydataset.mydataset import mydataset

  d = mydataset('lip')
  res = d.roidb
  from IPython import embed;

  embed()
