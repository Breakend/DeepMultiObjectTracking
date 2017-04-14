import os
import xml.etree.ElementTree as ET

import chainer
import cv2
import numpy as np

names = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)


class VOCDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, sets, size):
        self.root = root
        self.size = size

        self.images = list()
        for year, name in sets:
            root = os.path.join(self.root, 'VOC' + year)
            for line in open(
                    os.path.join(root, 'ImageSets', 'Main', name + '.txt')):
                self.images.append((root, line.strip()))

    def __len__(self):
        return len(self.images)

    def name(self, i):
        return self.images[i][1]

    def image(self, i):
        return cv2.imread(
            os.path.join(
                self.images[i][0], 'JPEGImages', self.images[i][1] + '.jpg'),
            cv2.IMREAD_COLOR)

    def get_example(self, i):
        img = self.image(i)
        h, w, _ = img.shape
        img = cv2.resize(img, (self.size, self.size))
        input_height, input_width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        tree = ET.parse(os.path.join(
            self.images[i][0], 'Annotations', self.images[i][1] + '.xml'))
        t = []
        for child in tree.getroot():
            if not child.tag == 'object':
                continue
            bndbox = child.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            class_id = names.index(child.find('name').text)
            one_hot_label = np.zeros(len(names))
            one_hot_label[class_id] = 1
            t.append({
                "x": (xmin + xmax) / (2 * w),
                "y": (ymin + ymax) / (2 * h),
                "w": (xmax - xmin) / w,
                "h": (ymax - ymin) / h,
                "label": class_id,
                "one_hot_label": one_hot_label
            })
        return img, t
