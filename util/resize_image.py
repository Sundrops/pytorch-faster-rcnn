import cv2

phase = 'train'
old_root = '/media/rgh/rgh-data/Dataset/Lip_t_d_zdf/'
new_root = '/media/rgh/rgh-data/Dataset/Lip_320/'
img_names = open('/media/rgh/rgh-data/Dataset/Lip_t_d_zdf/'+phase+'_full.txt', 'r')
for index, img_name in enumerate(img_names):
    img_name = img_name.strip()
    print index, img_name
    img = cv2.imread(old_root+'image/'+phase+'/'+img_name+'.jpg')
    img = cv2.resize(img,(320,320),interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(new_root+'image/'+phase+'_full/'+img_name+'.jpg',img)

    label = cv2.imread(old_root + 'label/' + phase + '/' + img_name + '.png')
    label = cv2.resize(label, (320, 320), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(new_root + 'label/' + phase + '_full/' + img_name + '.png', label)

    #break