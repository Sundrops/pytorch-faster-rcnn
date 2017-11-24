import cv2
import cPickle as pickle
import numpy as np
# b g r
colorbar=[
    (230,230,230),(0,215,255),(49,49,80),(255,0,51),(49,251,2),
    (212,255,141),(255,0,160),(255,204,0),(248,255,191),(185,182,255),
    (121,122,180),(57,160,202)
]
#fr = open('/home/rgh/PycharmProjects/pytorch-faster-rcnn/output/vgg16/lip_val/finetuning_coco/vgg16_faster_rcnn_iter_70000/detections.pkl')
fr = open('/home/rgh/PycharmProjects/pytorch-faster-rcnn/output/vgg16/lip_val/default/vgg16_faster_rcnn_iter_70000/detections.pkl')
all_boxes = pickle.load(fr)
img_names = open('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/val.txt','r')
for im_ind, img_name in enumerate(img_names):
    img_name = img_name.strip()
    print im_ind, img_name
    img = cv2.imread('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/image/val/'+img_name+'.jpg')
    img1 = img.copy()
    for cls_ind in range(1, 12):
        dets = all_boxes[cls_ind][im_ind]
        if dets == []:
            continue
        for k in range(dets.shape[0]):
            if dets[k, -1] > 0.1:
                #print(im_ind,cls_ind, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,dets[k, 2] + 1, dets[k, 3] + 1)
                img1 = cv2.rectangle(img1,(int(dets[k, 0]+1), int(dets[k, 1] + 1)),
                                    (int(dets[k, 2] + 1), int(dets[k, 3] + 1)),colorbar[cls_ind], thickness=2)
        #break

    # gt_bbox = open('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/bbox/val/'+img_name+'.txt','r')
    # for index, line in enumerate(gt_bbox):
    #   line = line.strip()
    #   if line != '':
    #     x1, y1, x2, y2 = line.split(' ')
    #     # Make pixel indexes 0-based
    #     x1 = int(x1)# - 1 #if float(x1) > 1 else float(x1)
    #     y1 = int(y1)# - 1 #if float(y1) > 1 else float(y1)
    #     x2 = int(x2)# - 1
    #     y2 = int(y2)# - 1
    #     img1 = cv2.rectangle(img1, (x1,y1),
    #                     (x2, y2), (0, 255, 0))

    # cv2.imwrite('/home/rgh/PycharmProjects/pytorch-faster-rcnn/output/vgg16/lip_val/viz/' + img_name + '.jpg', img)
    #cv2.imwrite('/home/rgh/PycharmProjects/pytorch-faster-rcnn/output/vgg16/lip_val/viz/' + img_name + '_gt.jpg', img1)
    #break
    img = cv2.resize(img,(121,241), interpolation=cv2.INTER_LINEAR)
    img1 = cv2.resize(img1, (121, 241), interpolation=cv2.INTER_LINEAR)
    deeplab_viz = cv2.imread('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/deeplab_viz/val/'+img_name+'.png')
    gt_viz = cv2.imread('/media/rgh/rgh-data/Dataset/CVPR2018/Lip/gt_viz/val/' + img_name + '.png')
    line = np.zeros((img.shape[0], 5, 3))
    final = np.concatenate((img, line, img1, line, deeplab_viz, line, gt_viz), 1)
    cv2.imwrite('/home/rgh/PycharmProjects/pytorch-faster-rcnn/output/vgg16/lip_val/viz_0.1/' + img_name + '.jpg', final)
fr.close()