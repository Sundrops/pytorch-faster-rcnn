import os
import numpy as np

labels ={}
root = '/home/rgh/Documents/Tencent Files/570070539/FileRecv/train_val_all_cluster/class_all/'
bbox_dir = '/media/rgh/rgh-data/Dataset/Lip_320/bbox/'
for i in range(1,20):
    txt = open(root+str(i)+'.txt', 'r')
    for index, line in enumerate(txt):
        if index == 0:
            continue  # pass first line
        line = line.strip()
        name, label = line.split(' ')
        if name not in labels.keys():
            labels[name] = []
            labels[name].append(str(i) + ' ' + label)
        else:
            labels[name].append(str(i) + ' ' + label)
print len(labels)
sub_category_dir = '/media/rgh/rgh-data/Dataset/Lip_320/sub_category_3/'
phases = ['train_full', 'val']
for phase in phases:
    names = open('/media/rgh/rgh-data/Dataset/Lip_320/'+phase+'.txt','r')
    for name in names:
        name = name.strip()
        sub_txt = open(sub_category_dir+phase+'/'+name+'.txt', 'w')
        bbox_txt = open()
        for value in labels[name]:
            sub_txt.write(value + '\n')
