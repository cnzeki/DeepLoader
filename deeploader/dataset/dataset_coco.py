# coding=utf-8
import os
import sys
import numpy as np
from pycocotools.coco import COCO
from deeploader.dataset.dataset_cls import ClassifyDataset
import cv2

class COCODataset(ClassifyDataset):
    def __init__(self, dataroot, subset="val", year="2014", task="instances", name='COCO'):
        ClassifyDataset.__init__(self, name="{}{}{}".format(name, subset, year))
        self.dataset_dir = dataroot
        self.coco = COCO("{}/annotations/{}_{}{}.json".format(dataroot,task, subset, year))
        self.images_dir = os.path.join(dataroot, "images/%s%s" % (subset, year))
        # cat_id 1-90
        self.cats = self.coco.dataset['categories']
        # cat_id to zero-based idx
        cat2idx = {}
        self.class_names = []
        for idx, cat in enumerate(self.cats):
            cat2idx[cat['id']] = idx
            self.class_names.append(cat['name'])
        self.cat2idx = cat2idx
        # image
        self.ids = list(self.coco.imgs.keys())

    def classNames(self):
        return self.class_names

    def label2catId(self, label):
        return self.cats[label]['id']

    def catId2label(self, catId):
        return self.cat2idx[catId]

    def size(self):
        return len(self.ids)

    def numOfClass(self):
        return len(self.cats)

    @property
    def class_nums(self):
        return self.numOfClass()

    def getClassData(self, classId):
        catId = self.cat2idx[catId]
        imgIds = self.coco.getImgIds(catIds=catId)
        return imgIds

    def getData(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_anns = coco.loadAnns(ann_ids)
        imgIds = coco.getImgIds(imgIds=img_id)
        imgInfo = coco.loadImgs(imgIds)[0]
        img_path = os.path.join(self.images_dir, imgInfo['file_name'])
        img = cv2.imread(img_path)

        ret = dict(path=img_path, img=img, annos = img_anns, info=imgInfo)
        return ret

    def viz(self, ret, saveTo=None):
        from skimage import io
        from matplotlib import pyplot as plt
        # I = io.imread(ret['path'])
        I = ret['img']
        I[:] = 0
        plt.axis('off')
        plt.imshow(I)
        self.coco.showAnns(ret['annos'])
        if saveTo:
            plt.savefig(saveTo+'/'+ret['info']['file_name'])
            plt.clf()
            return
        plt.show()
        #plt.waitforbuttonpress()
