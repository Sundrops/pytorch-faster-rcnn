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
import cv2
import numpy as np
import torch


def get_all_boxes():
  label_dir = '/media/rgh/rgh-data/PycharmProjects/cvpr2018/deeplab/deeplab_result_40epoch/outLabel_lip/'
  names = open('/media/rgh/rgh-data/Dataset/Lip_320/val.txt','r')
  class_nums = 20
  img_nums = 1914
  all_boxes = [[[] for _ in range(img_nums)]
               for _ in range(class_nums)]
  #print(len(names))
  for index, name in enumerate(names):
    name = name.strip()
    print(index, ' ', name)
    label = cv2.imread(label_dir + name+'.png', cv2.IMREAD_GRAYSCALE)
    label = cv2.resize(label, (320, 320), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('/media/rgh/rgh-data/PycharmProjects/cvpr2018/deeplab/deeplab_result_40epoch/320/'+ name+'.png'
                ,label)
    # 1 - 19
    for i in range(1, class_nums):
      label_part = label == i
      label_part = label_part - 0
      if not 1 in label_part:
        continue
      h = label_part.shape[0]
      w = label_part.shape[1]
      for j in range(h):
        if 1 in label_part[j]:
          y1 = j
          break
      for j in range(h):
        if 1 in label_part[h - 1 - j]:
          y2 = h - 1 - j
          break
      for j in range(w):
        if 1 in label_part[:, j]:
          x1 = j
          break
      for j in range(w):
        if 1 in label_part[:, w - 1 - j]:
          x2 = w - 1 - j
          break
      all_boxes[i][index] = np.array([[x1, y1, x2, y2, 1]])
  return all_boxes
if __name__ == '__main__':

    #all_boxes = get_all_boxes()
    det_file = os.path.join('/media/rgh/rgh-data/PycharmProjects/cvpr2018/deeplab', 'detections.pkl')
    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    all_boxes = pickle.load(open(det_file, 'rb'))
    print(all_boxes[:][1])
    print(len(all_boxes))
    print(len(all_boxes[1]))
    print(all_boxes[2][1])
    # imdb_name = 'Lip_320_val'
    # imdb = get_imdb(imdb_name)
    # print('Evaluating detections')
    # imdb.evaluate_detections(all_boxes, '/media/rgh/rgh-data/PycharmProjects/cvpr2018/temp/')
