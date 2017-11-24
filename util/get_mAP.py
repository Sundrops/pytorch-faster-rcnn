# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import pickle
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

import torch

# cfg_from_file('/media/rgh/rgh-data/PycharmProjects/cvpr2018/experiments/cfgs/vgg16.yml')
# det_file = '/media/rgh/rgh-data/PycharmProjects/cvpr2018/output/vgg16/Lip_320_val/' \
#            'default/rgh' \
#            '/detections.pkl'

if __name__ == '__main__':
  imdb_name = 'Lip_320_val'
  det_file= '/media/rgh/rgh-data/PycharmProjects/cvpr2018/zdf/pred_bbox_leftleg(13).pkl'

  #det_file = '/media/rgh/rgh-data/PycharmProjects/cvpr2018/output/vgg16/Lip_320_val/default/vgg16_faster_rcnn_iter_70000/detections.pkl'
  imdb = get_imdb(imdb_name)
  all_boxes = pickle.load(open(det_file, 'rb'))
  #print(all_boxes[:][1])
  #print(len(all_boxes))
  #print(len(all_boxes[1]))
  #print(all_boxes[1][1])
  # print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, '/media/rgh/rgh-data/PycharmProjects/cvpr2018/temp/')
