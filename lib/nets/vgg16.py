# coding=utf-8
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.network import Network
from model.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models

class vgg16(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._net_conv_channels = cfg.FC6_IN_CHANNEL
    self._fc7_channels = cfg.FC7_OUT_CHANNEL

  def _init_head_tail(self):
    self.vgg = models.vgg16()
    if cfg.LIGHT_RCNN:
      # 去除一个隐层的,没有dropout
      self.vgg.classifier = nn.Sequential(
        nn.Linear(cfg.FC6_IN_CHANNEL * 7 * 7, cfg.FC7_OUT_CHANNEL),
        nn.ReLU(True),
        # nn.Linear(4096, 4096),
        # nn.ReLU(True),
        # nn.Dropout(),
      )
    else:
      print('nn.Linear(512 * 7 * 7, 4096)')
      # Remove fc8
      self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier._modules.values())[:-1])
      # self.vgg.classifier = nn.Sequential(
      #   nn.Linear(cfg.FC6_IN_CHANNEL * 7 * 7, 4096),
      #   nn.ReLU(True),
      #   nn.Dropout(),
      #   nn.Linear(4096, 4096),
      #   nn.ReLU(True),
      #   nn.Dropout(),
      # )
    if cfg.FIX_FEAT:
      # Fix all layers
      for layer in range(30):
        for p in self.vgg.features[layer].parameters(): p.requires_grad = False
      for p in self.vgg.classifier[0].parameters(): p.requires_grad = False  # nn.Linear(cfg.FC6_IN_CHANNEL * 7 * 7, 4096)
      for p in self.vgg.classifier[3].parameters(): p.requires_grad = False  # nn.Linear(4096, 4096),
    else:
      # Fix the layers before conv3:
      for layer in range(10):
        for p in self.vgg.features[layer].parameters(): p.requires_grad = False
    # not using the last maxpool layer
    self._layers['head'] = nn.Sequential(*list(self.vgg.features._modules.values())[:-1])

  def _image_to_head(self):
    net_conv = self._layers['head'](self._image)
    self._act_summaries['conv'] = net_conv

    return net_conv

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.vgg.classifier(pool5_flat)

    return fc7

  def load_pretrained_cnn(self, state_dict):
    #self.vgg.load_state_dict({k:v for k,v in state_dict.items() if k in self.vgg.state_dict()})
    print('vgg16')
    print('state_dict all keys: ', state_dict.keys())
    # only copy common items and has common shape
    if cfg.FIX_FEAT:
      model_dict = self.state_dict()
      state_dict = {k: v for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape}
      print('state_dict matched keys: ', state_dict.keys())
      model_dict.update(state_dict)
      self.load_state_dict(model_dict)
    else:
      model_dict = self.vgg.state_dict()
      state_dict = {k: v for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape}
      print('state_dict matched keys: ', state_dict.keys())
      model_dict.update(state_dict)
      self.vgg.load_state_dict(model_dict)

      '''
      if cfg.FC6_IN_CHANNEL == 512:
        model_dict = self.vgg.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
        print('state_dict matched keys: ', state_dict.keys())
        model_dict.update(state_dict)
        self.vgg.load_state_dict(model_dict)

      # roi crop_cat
      # only copy common Feature items and has common shape
      else:
        model_dict = self.vgg.state_dict()
        state_dict = {k: v for k, v in state_dict.items()
                      if (k.startswith('features') or k.startswith('classifier.3'))and k in model_dict and v.shape == model_dict[k].shape}
        print('state_dict matched keys: ', state_dict.keys())
        model_dict.update(state_dict)
        self.vgg.load_state_dict(model_dict)
      # vgg has alread init all layer with diff initialization
      '''