from confusion_matrix import ConfusionMatrix
import cv2
import numpy as np
matrix = ConfusionMatrix(size=20)
names = open('/media/rgh/rgh-data/Dataset/Lip_320/val.txt','r')
gt_label_dir = '/media/rgh/rgh-data/Dataset/Lip_320/label/val/'
predict_dir = '/'
for index, name in enumerate(names):
    name = name.strip()
    print index, name
    gt = cv2.imread(gt_label_dir+name+'.png',flags=cv2.IMREAD_GRAYSCALE)
    result = cv2.imread(predict_dir+name+'.png',flags=cv2.IMREAD_GRAYSCALE)
    matrix.update(gt, result)
print matrix.accuracy(), matrix.fg_accuracy(), matrix.avg_precision(), matrix.avg_recall(), matrix.avg_f1score()
print matrix.f1score()