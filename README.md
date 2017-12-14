# pytorch-faster-rcnn
fork自[ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
A pytorch implementation of faster RCNN detection framework based on Xinlei Chen's [tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn). Xinlei Chen's repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).


## 特别说明

此仓库是在[pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)基础下修改，网络结构加入了mask分支，实现了无fpn的mask rcnn. RoIAlign是用的类似[tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)的做法，和kaiming论文有一点点出入。
``experiments/cfgs/vgg16.yml``
- faster rcnn 复现mAP: 0.708
- 无fpn的mask rcnn
- [Light-Head R-CNN](https://arxiv.org/abs/1711.07264) 复现mAP: 0.711
```yml
# 为了融合全局特征，在roi pooling前加了类似U-Net的东西
ZDF_GAUSSIAN: False
ZDF: True
# 在原有分类基础上加了细分类，目的是通过multi-task提升原有的分类、检测和mask
SUB_CATEGORY: False
LOSS_SUB_CATEGORY_W: 0.5
# 这两个参数应对不同的POOLING_MODE
# pyramid_crop_sum金字塔roi(1,1.5,2)
# pyramid_crop金字塔roi cat后降维
# 其他模式可能使最终的输出channel不为512，所以FC6_IN_CHANNEL要随之改动
POOLING_MODE: crop
# 是否做mask分支
DO_PARSING: True
# 我们的训练是分两步，一步是先把检测(DO_PARSING: False)训练好
# 固定检测的所有参数，只训练mask分支
# 为了方便，训练好检测好把最终模型改名为vgg16，放到data/imagenet_weights，然后设置FIX_FEAT: True
FIX_FEAT: True
# light rcnn 输出的feature是k*7*7，此处k设置为10 10*7*7=490
# 且去掉了一个fc的隐层，只留一个2048的fc隐层(无dropout)
# 且large kernel cmid设置为128
LIGHT_RCNN: True
FC6_IN_CHANNEL: 490
FC7_OUT_CHANNEL: 2048
```

```shell
# pascal_voc 数据集 vgg16 网络结构 default 标签
 ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16 default
```

### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by [pytorch-vgg](https://github.com/jcjohnson/pytorch-vgg.git) and [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) (the ones with caffe in the name), you can download the pre-trained models and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   python # open python in terminal and run the following Python code
   ```
   ```Python
   import torch
   from torch.utils.model_zoo import load_url
   from torchvision import models

   sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
   sd['classifier.0.weight'] = sd['classifier.1.weight']
   sd['classifier.0.bias'] = sd['classifier.1.bias']
   del sd['classifier.1.weight']
   del sd['classifier.1.bias']

   sd['classifier.3.weight'] = sd['classifier.4.weight']
   sd['classifier.3.bias'] = sd['classifier.4.bias']
   del sd['classifier.4.weight']
   del sd['classifier.4.bias']

   torch.save(sd, "vgg16.pth")
   ```
   ```Shell
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   # download from my gdrive (link in pytorch-resnet)
   mv resnet101-caffe.pth res101.pth
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET] [TAG]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16 default
  ./experiments/scripts/train_faster_rcnn.sh 1 coco res101 yourtag
  ```
  **Note**: Please double check you have deleted soft link to the pre-trained models before training. If you find NaNs during training, please refer to [Issue 86](https://github.com/endernewton/tf-faster-rcnn/issues/86). Also if you want to have multi-gpu support, check out [Issue 121](https://github.com/endernewton/tf-faster-rcnn/issues/121).

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_faster_rcnn.sh 1 coco res101
  ```

5. You can use ``tools/reval.sh`` for re-evaluation


By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```
