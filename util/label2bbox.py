import os
import cv2
import numpy as np
import cv2
import os
from PIL import Image
import numpy as np

phase = 'train_full'
os.chdir('/media/rgh/rgh-data/Dataset/Lip_320/label/' + phase + '/')
img_dir = '/media/rgh/rgh-data/Dataset/Lip_320/image/' + phase + '/'
bbox_dir = '/media/rgh/rgh-data/Dataset/Lip_320/bbox/' + phase + '/'

class_nums = 20
for index, name in enumerate(os.listdir('.')):
    # continue
    print name
    label = cv2.imread(name, flags=cv2.IMREAD_GRAYSCALE)
    name_pre = name[:name.find('.')]
    img = cv2.imread(img_dir + name_pre + '.jpg')
    bbox_txt = open(bbox_dir + name_pre + '.txt', 'w')
    # 1 - 19
    for i in range(1, class_nums):
        label_part = label == i
        label_part = label_part - 0
        if not 1 in label_part:
            if i != class_nums - 1:
                bbox_txt.write('\n')
            else:
                bbox_txt.write('')
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
        if i != class_nums - 1:
            bbox_txt.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')
        else:
            bbox_txt.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2))

        # target_img = img[y1:y2 + 1, x1:x2 + 1]
        # target_img = cv2.resize(target_img, (128, 128), interpolation=cv2.INTER_LINEAR)
        # target_label = label_part[y1:y2 + 1, x1:x2 + 1]
        # target_label = cv2.resize(target_label, (128, 128), interpolation=cv2.INTER_NEAREST)
    #break
