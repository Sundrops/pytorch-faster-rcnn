# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
import pickle
import torchvision.models as models

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# a =  torch.randn((1,2,9,9))
# print a
#
# b = torch.nn.functional.adaptive_max_pool2d(a, 3)
# print b
# b = Variable(torch.ones((3,2)), requires_grad=False)
# b[b<0] = 0
# print(b)
# model = torch.load('/home/rgh/PycharmProjects/pytorch-faster-rcnn/data/imagenet_weights/vgg16-00b39a1b.pth')
# print(model.keys())
# model = models.vgg16()
#
# b = torch.ones((3,2))
# c = b[:, 0]
# # c = b.new(b.size(0), 1) #.zero_()
# print(b)
# print(c.size())
# clss = torch.from_numpy(np.array([0,1,2]))
# bbox_targets = clss.new(clss.numel(), 4 * 3).zero_()
# bbox_target_data = torch.from_numpy(np.array([[-1,-2,-3,-4],[-1,-2,-3,-4],[-1,-2,-3,-4]]))
# inds = torch.from_numpy(np.array([1,2]))
#
# clss = clss[inds].contiguous().view(-1,1)  # (n, 1)
# print(clss)
# dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)  # (n, 4) [ [1,1,1,1],[3,3,3,3],[4,4,4,4] ]
# print(dim1_inds)
# dim2_inds = torch.cat([4*clss, 4*clss+1, 4*clss+2, 4*clss+3], 1).long()
# print(dim2_inds)
# bbox_targets[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 0:]
#
# scores = np.array([[1,2],[3,4],[5,6]])
# mask_target_channel = scores.argmax(1)
# print(mask_target_channel.shape)
# img = np.array([0.2,0.5])
# cv2.imwrite('/media/rgh/rgh-data/PycharmProjects/cvpr2018/temp/mask/test.png',img)
# result = np.zeros((3,3))
#
# mask_a = np.array([[1,0],[3,4]])
# index = result[0:2,0:2]==0
# result[0:2,0:2][index] = mask_a[index]
#
# mask_b = np.array([[100,101],[102,103]])
# index = result[0:2,0:2]==0
# result[0:2,0:2][index] = mask_b[index]
# print(result)
# print(bbox_targets)
# a = torch.from_numpy(np.array([[1,2,3],[4,8,6]]))
# b = a[0:1,:]
# print(b)
# b = a.new(2, 1).zero_()
# for i in range(2):
#     b[i,:] = a[i,1]
# b[0][0] = 10
# print(a)
# x = np.ones((3,2))
# y = np.ones(3)
# print(x+y)
# overlaps = torch.from_numpy(np.array([[1,2,3],[4,8,6]]))
# gt_boxes = torch.from_numpy(np.array([[0,0,0,0,-1],[0,0,0,0,-2],[0,0,0,0,-3]]))
# max_overlaps, gt_assignment = overlaps.max(1)
# print(max_overlaps,gt_assignment)
# labels = gt_boxes[gt_assignment, [4]]
# print(labels)
# for key, value in dict(model.named_parameters()).items():
#     print(key)
# obj2 = pickle.load(open(os.path.join('/home/rgh/PycharmProjects/pytorch-faster-rcnn/output/vgg16/lip_val/global_cat/vgg16_faster_rcnn_iter_70000',
#                        'hair_pr.pkl'), 'rb'))
# print(obj2)

# a = torch.ones((256,512,37,39))
#
# b = torch.zeros((256,37,39))
# print(b.size())
# a = Variable(a)
# print(a.size())
# b = Variable(b)
# print(a*b)
# all_boxes = [[[3] for _ in range(2)]
#          for _ in range(3)]
# all_boxes[0][0] = 10
# print(all_boxes)
# print(all_boxes[0][0][1])
# img = cv2.imread('/home/rgh/Pictures/temp/t.jpg')
# img = np.array(img)
# img = np.array([img])
# b = torch.from_numpy(img)
# b = b.float()
# b = b.permute(0, 3, 1, 2)
# print(b.size())
# # b has the size (1, 3, 360, 640)
# flow = torch.zeros(1, 420, 1002 , 2)
# b = Variable(b)
# flow = Variable(flow)
# out = F.grid_sample(b, flow)
# print(out.data.shape)
# img = out.data.numpy()
# print(img.shape)
# img = img.squeeze().transpose(1,2,0)
# cv2.imshow('g',img)
# cv2.waitKey()



# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

# x = Variable(torch.ones((2,2)), requires_grad=True)
#
# z = x * 2
# z[0,0] = 0
# z.register_hook(save_grad('z'))
# t = z+2
#
# t.backward(torch.Tensor([[10]]))
# print(x.grad,grads['z'])

# x = Variable(torch.ones(2, 2), requires_grad = True)
# w = Variable(torch.ones(2, 2), requires_grad = True)
# z = x*w -100
# loss = z.sum()
# loss.backward()  # loss.backward(torch.Tensor([1]))
#
# print(x.grad)
# print(w.grad)
# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook
#
# x = Variable(torch.ones((2,2)), requires_grad=True)
# y = Variable(torch.ones((1,2)), requires_grad=True)
# t = Variable(torch.ones((1,2)), requires_grad=True)
# t = x[0,:] + y
# t.register_hook(save_grad('t'))
# z = torch.cat((y, t),1)
# z = torch.cat((y, z),1)
# z.backward(torch.Tensor([[1, 2, 3, 4, 1,1]]))
# print(z, x.grad, y.grad, t.grad, grads['t'])
#
# z = x[0,0]
# z.backward(torch.Tensor([[10]]))
# print(z, x.grad, y.grad, t.grad)
# a = torch.ones((3,2))
# a[0,0] = 3
# print(a==3)

# a = np.array([[1,2],[3,4]])
# b = np.zeros((2,2,3))
# mask = a == 1
# c = b[mask]
# c[0][0] = -100
# print (c)
# print (b)
# d = a[mask]
# d [0] = -100
# print (d)
# print (a)
# a = np.array([[1,2],[3,4]])
# mask = a == 1
# b = a[mask]
# b = -100
# print (b)
# print (a)
# a[mask] = -100
# print (a)
# img_names = open('/media/rgh/rgh-data/Dataset/Lip_t_d_zdf/train_full.txt', 'r')
# total = []
# for name in img_names:
#     name = name.strip()
#     if name not in total:
#         total.append(name)
#     else:
#         print(name)
# print (len(total))
class Vars(object):

    def __init__(self):
        self.count = 0
        self.defs = {}
        self.lookup = {}

    def add(self, *v):
        name = self.lookup.get(v, None)
        print (v)
        if name is None:
            if v[0] == '+':
                if v[1] == 0:
                    return v[2]
                elif v[2] == 0:
                    return v[1]
            elif v[0] == '*':
                if v[1] == 1:
                    return v[2]
                elif v[2] == 1:
                    return v[1]
                elif v[1] == 0:
                    return 0
                elif v[2] == 0:
                    return 0

            self.count += 1
            name = "v" + str(self.count)
            self.defs[name] = v
            self.lookup[v] = name

        return name

    def __getitem__(self, name):
        return self.defs[name]

    def __iter__(self):
        return self.defs.iteritems()


def diff(vars, acc, v, w):
    if v == w:
        return acc

    v = vars[v]
    if v[0] == 'input':
        return 0
    elif v[0] == "sin":
        return diff(vars, vars.add("*", acc, vars.add("cos", v[1])), v[1], w)
    elif v[0] == '+':
        gx = diff(vars, acc, v[1], w)
        gy = diff(vars, acc, v[2], w)
        return vars.add("+", gx, gy)
    elif v[0] == '*':
        gx = diff(vars, vars.add("*", v[2], acc), v[1], w)
        gy = diff(vars, vars.add("*", v[1], acc), v[2], w)
        return vars.add("+", gx, gy)

    raise NotImplementedError


def autodiff(vars, v, *wrt):
    return tuple(diff(vars, 1, v, w) for w in wrt)


# z = (sin x) + (x * y)

vars = Vars()

x = vars.add("input",1)
#a= raw_input()
y = vars.add("input",2)
z = vars.add("+", vars.add("*",x,y),vars.add("sin",x))

a= raw_input()
print (autodiff(vars, z, x, y))

for k, v in vars:
    print (k, v)