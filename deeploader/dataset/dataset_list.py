# -*- coding:utf-8 -*-  
from __future__ import print_function
import scipy
import numpy as np
import os
import sys
import time
import random
import math
import cv2
from tqdm import tqdm
from deeploader.dataset.dataset_cls import ClassifyDataset

def read_img(p):
    path, y = p[0], p[1]
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x, y
        
class FileListReader(ClassifyDataset):
    def __init__(self, datadir, list_path, balance = False, name='--'):
        ClassifyDataset.__init__(self, name)
        self.datadir = datadir
        self.celeb = []
        self.dict = {}
        self.list = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                self.add(line)
        # sort label
        keys = set()
        bar = tqdm(total=len(self.list))
        for index in range(len(self.list)):
            item = self.list[index]
            label = item[1]
            bar.update(1)
            # add to dict[label]
            #if not label in self.dict.keys():
            if not label in keys:
                self.dict[label] = []
                keys.add(label)
            self.dict[label].append(index)
        bar.close()
        # update celeb
        #print(self.dict)
        keys = list(self.dict.keys())
        keys.sort()
        self.celeb = keys

        
    def add(self, line):
        line = line.strip()
        if line.find(',') > 0:
            segs = line.split(',')
        else:
            segs = line.split()
        #print(segs)
        if len(segs) != 2 :
            return False
        rel_path = segs[0]
        label = int(segs[1])
        item = (rel_path, label)
        # add to list
        self.list.append(item)

        
    def size(self):
        return len(self.list)

    def numOfClass(self):
        return len(self.celeb)

    def getData(self, index):
        item = self.list[index]
        rel_path = item[0]
        path = os.path.join(self.datadir, rel_path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return {'img':img, 'label':item[1], 'path': rel_path }
        
        
    def getClassData(self, classId):
        return self.dict[self.celeb[classId]]
        
if __name__ == '__main__':
    reader = FileDirReader('E:/data/faces_glintasia_snap80')
    pass
    