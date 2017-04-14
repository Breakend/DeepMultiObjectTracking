#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json

import chainer
from chainer import cuda
from chainer import serializers

from lib import rect
from voc import VOCDataset
from yolov2 import *


def non_maximum_suppression(boxes, conf, nms_threshold, conf_threshold):
    for cls in range(conf.shape[1]):
        selected = np.zeros((conf.shape[0],), dtype=bool)
        for i in conf[:, cls].argsort()[::-1]:
            if conf[i, cls] < conf_threshold:
                break
            box = rect.Rect.LTRB(*boxes[i])
            iou = rect.matrix_iou(
                boxes[np.newaxis, i],
                boxes[selected])
            if (iou >= nms_threshold).any():
                continue
            selected[i] = True
            yield box, cls, conf[i, cls]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dtype', choices=['voc', 'coco'])
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--test', action='append')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--loaderjob', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('model')
    args = parser.parse_args()

    config = json.load(open("config.json", 'r'))
    im_size = 416
    classes = config[args.dtype]["classes"]

    # load model
    print("loading initial model...")
    yolov2 = YOLOv2(config[args.dtype])
    serializers.load_npz(args.model, yolov2)
    model = YOLOv2Predictor(yolov2)
    model.predictor.train = False
    model.predictor.finetune = False

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np


    def dump_result(name, boxes, prob, conf, w, h, classes):
        if xp is not np:
            boxes = xp.asnumpy(boxes)
            prob = xp.asnumpy(prob)
            conf = xp.asnumpy(conf)

        boxes = boxes.reshape((-1, 4)).clip(min=0.0, max=1.0)
        prob = prob.reshape((-1, 20))
        conf = conf.reshape((-1, 1))
        prob = prob * conf
        nms = non_maximum_suppression(boxes, prob, 0.45, 0.005)
        for box, cls, score in nms:
            filename = 'result/comp4_det_test_{:s}.txt'.format(classes[cls])
            with open(filename, mode='a') as f:
                print(
                    name, score, box.left * w, box.top * h, box.right * w,
                                 box.bottom * h, file=f)


    dataset = VOCDataset(args.root, [t.split('-') for t in args.test], im_size)

    info = list()
    batch = list()
    for i in range(len(dataset)):
        img = dataset.image(i)
        h, w, _ = img.shape
        info.append((dataset.name(i), (w, h)))
        img = cv2.resize(img, (im_size, im_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        batch.append(img)

        if len(batch) == args.batchsize:
            boxes, conf, prob = model.predict(
                Variable(xp.array(batch), volatile=True))
            for i, (name, (w, h)) in enumerate(info):
                dump_result(name, boxes.data[i], prob.data[i], conf.data[i], w,
                            h, classes)
            info = list()
            batch = list()

    if len(batch) > 0:
        boxes, conf, prob = model.predict(
            Variable(xp.array(batch), volatile=True))
        for i, (name, size) in enumerate(info):
            dump_result(name, boxes.data[i], prob.data[i], conf.data[i], w, h,
                        classes)
