#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json

import chainer
from chainer import serializers, optimizers, Variable

from voc import VOCDataset
from yolov2 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dtype', choices=['voc', 'coco'])
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--train', action='append')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--loaderjob', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--init')
    parser.add_argument('--resume')
    args = parser.parse_args()

    config = json.load(open("config.json", 'r'))
    size = 416

    # hyper parameters
    max_batches = 45000
    learning_rate = 1e-4
    learning_schedules = {
        "0": 1e-4,
        "100": 1e-3,
        "25000": 1e-4,
        "35000": 1e-5
    }
    backup_file = "%s/backup.model" % (args.out)

    momentum = 0.9
    weight_decay = 0.005

    # load model
    print("loading initial model...")
    yolov2 = YOLOv2(config[args.dtype])
    model = YOLOv2Predictor(yolov2)
    if args.init:
        serializers.load_hdf5(args.init, yolov2)

    model.predictor.train = True
    model.predictor.finetune = False

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    train = VOCDataset(args.root, [t.split('-') for t in args.train], size)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)

    optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
    optimizer.use_cleargrads()
    optimizer.setup(model)

    # start to train
    print("start training")
    for batch in range(max_batches):
        if str(batch) in learning_schedules:
            optimizer.lr = learning_schedules[str(batch)]

        # generate sample
        batch_list = train_iter.__next__()
        x = []
        t = []
        for j in range(len(batch_list)):
            x.append(batch_list[j][0])
            t.append(batch_list[j][1])
        x = np.array(x)
        x = Variable(x)
        x.to_gpu()

        # forward
        loss = model(x, t)
        print("batch: %d\tlearning rate: %f\tloss: %f" % (
            batch, optimizer.lr, loss.data))
        # backward and optimize
        optimizer.zero_grads()
        loss.backward()
        optimizer.update()

        # save model
        if (batch + 1) % 1000 == 0:
            model_file = "%s/%s.model" % (args.out, batch + 1)
            print("saving model to %s" % (model_file))
            serializers.save_hdf5(model_file, model)

    print("saving model to %s/yolov2_final.model" % (args.out))
    serializers.save_hdf5("%s/yolov2_final.model" % (args.out), model)
