#!/bin/bash
#!/usr/bin/env python
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
TAG=$4
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  Lip_t_d)
    TRAIN_IMDB="Lip_t_d_train"
    TEST_IMDB="Lip_t_d_val"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  Lip_320_full)
    TRAIN_IMDB="Lip_320_train_full"
    TEST_IMDB="Lip_320_val"
    STEPSIZE="[100000]"
    ITERS=280000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  Lip_320)
    TRAIN_IMDB="Lip_320_train"
    TEST_IMDB="Lip_320_val"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${TAG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
#if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
#  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
#else
#  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.pth
#fi
NET_FINAL=output/${NET}/${TRAIN_IMDB}/${TAG}/${NET}_faster_rcnn_iter_${ITERS}.pth
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.pth \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${TAG} \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.pth \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${TAG} \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_faster_rcnn.sh $@
