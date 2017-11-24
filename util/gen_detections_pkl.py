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

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

import torch


if __name__ == '__main__':
    imdb_name = 'Lip_320_train'
    cfg_from_file('/media/rgh/rgh-data/PycharmProjects/cvpr2018/experiments/cfgs/vgg16.yml')
    tag = 'default/rgh'
    model = '/media/rgh/rgh-data/PycharmProjects/cvpr2018/output/vgg16/Lip_320_train/default/' \
                       + 'vgg16_faster_rcnn_iter_70000.pth'
    cfg.ANCHOR_SCALES = [4, 8, 16, 32]
    cfg.ANCHOR_RATIOS = [0.5, 1, 2]
    #cfg.HAS_PARSING_LABEL = False
    cfg.POOLING_MODE = 'crop'
    cfg.FC6_IN_CHANNEL = 512
    cfg.TEST.CLEAN_PRE_RESULT = True
    print('Using config:')
    pprint.pprint(cfg)
    imdb = get_imdb(imdb_name)
    net = vgg16()
    net.create_architecture(imdb.num_classes, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)
    net.eval()
    net.cuda()

    model_dict = torch.load(model)
    print(model_dict.keys())
    print(model_dict['vgg.classifier.0.weight'].shape)
    net.load_state_dict(model_dict)

    test_net(net, imdb, tag, max_per_image=100, clean_pre_result=cfg.TEST.CLEAN_PRE_RESULT)
