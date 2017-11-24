import cv2
import os
import shutil
labels ={}

root= '/media/rgh/rgh-data/Dataset/CVPR2018/zdf/sub_cls/'
img_dir = '/media/rgh/rgh-data/Dataset/Lip_t_d_zdf/image/train/'
label_dir = '/media/rgh/rgh-data/Dataset/Lip_t_d_zdf/label/train/'
for i in range(1,20):
    txt = open('/home/rgh/Documents/Tencent Files/570070539/FileRecv/lip_cluster5/class_all/'+str(i)+'.txt', 'r')
    for index, line in enumerate(txt):
        if index == 0:
            continue  # pass first line
        line = line.strip()
        name, label = line.split(' ')
        if not os.path.exists(root+str(i)+'/'):
            os.mkdir(root+str(i)+'/')
        dst = root+str(i)+'/'+label+'/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        img = cv2.imread(img_dir+name+'.jpg')
        label = cv2.imread(label_dir+name+'.png',flags=cv2.IMREAD_GRAYSCALE)
        mask = label!= int(i)
        img[mask] = 0
        cv2.imwrite(dst+name+'.jpg',img)
        if index == 200:
            break
        #break
    #break


