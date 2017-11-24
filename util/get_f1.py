from confusion_matrix import ConfusionMatrix
from util.common import color
import cv2
import numpy as np
matrix = ConfusionMatrix(size=20)
names = open('/media/rgh/rgh-data/Dataset/Lip_320/val.txt','r')
gt_label_dir = '/media/rgh/rgh-data/Dataset/Lip_320/label/val/'
img_dir = '/media/rgh/rgh-data/Dataset/Lip_320/image/val/'
result_dir = '/media/rgh/rgh-data/PycharmProjects/cvpr2018/output/vgg16/Lip_320_val/parsing_fix/vgg16_faster_rcnn_iter_70000/parsing/'
for index, name in enumerate(names):
    name = name.strip()
    print index, name
    img = cv2.imread(img_dir + name + '.jpg')
    gt = cv2.imread(gt_label_dir+name+'.png',flags=cv2.IMREAD_GRAYSCALE)
    #print img.shape
    gt_viz = color(gt)
    #print gt_viz.shape
    result = cv2.imread(result_dir+name+'.png',flags=cv2.IMREAD_GRAYSCALE)
    result_viz = color(result)
    #print result_viz.shape
    line = np.zeros((gt.shape[0], 5, 3))

    fuse = np.concatenate((img, line, result_viz, line, gt_viz), 1)
    cv2.imwrite('/media/rgh/rgh-data/PycharmProjects/cvpr2018/output/vgg16/Lip_320_val/parsing_fix/vgg16_faster_rcnn_iter_70000/parsing_viz/'
                +name+'.png',fuse)

    matrix.update(gt, result)
print matrix.accuracy(), matrix.fg_accuracy(), matrix.avg_precision(), matrix.avg_recall(), matrix.avg_f1score()
print matrix.f1score()