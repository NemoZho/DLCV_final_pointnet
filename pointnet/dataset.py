from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import random
import json
from plyfile import PlyData, PlyElement
import pandas as pd
from sklearn.preprocessing import normalize

VALID_CLASS_IDS_200 = [
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139, 140, 141, 145, 148, 154,
155, 156, 157, 159, 161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399, 408, 417,
488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191]

seed = 8787


def map_label(label):

    if label in VALID_CLASS_IDS_200:

        return VALID_CLASS_IDS_200.index(label) + 1
    else:
        return 0




class ScanNet200Dataset(data.Dataset):
    def __init__(self,
                 root,
                 filelist_txt,
                 split='train',
                 ratio=0.9,
                 data_augmentation=True):

        self.root = root
        self.filelist_temp = open(filelist_txt, "r").read().split("\n")
        random.seed(seed)
        self.filelist = random.choices(self.filelist_temp, k = int(len(self.filelist_temp) * ratio))

        if split == 'valid':
            self.filelist = [x for x in self.filelist_temp if x not in self.filelist]


    def __getitem__(self, index):
        fn = self.filelist[index]
        # print(fn)
        with open(os.path.join(self.root, fn), "rb") as f:
            plydata = PlyData.read(f)
        df = pd.DataFrame(plydata.elements[0].data)

        map_label_f = lambda t: map_label(t)
        vfunc = np.vectorize(map_label_f)

        point_set = df.loc[:, ['x', 'y', 'z', 'red', 'green', 'blue']].to_numpy().astype(np.float32)
        cls = vfunc(df.loc[:, ['label']].to_numpy().astype(np.int32))


        point_set[:, 0 : 3] = point_set[:, 0 : 3] - np.expand_dims(np.mean(point_set[:, 0 : 3], axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set[:, 0 : 3] ** 2, axis = 1)),0)
        point_set[:, 0 : 3] = point_set[:, 0 : 3] / dist #scale

        point_set[:, 3 : 6] = normalize(point_set[:, 3 : 6])
        # if self.data_augmentation:
        #     theta = np.random.uniform(0,np.pi*2)
        #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        #     point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
        #     point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        # seg = seg[choice]
        # point_set = np.pad(point_set, [(0, 00000 - point_set.shape[0]), (0, 0)], 'constant')
        # cls = np.pad(cls, [(0, 100000 - cls.shape[0]), (0, 0)], 'constant')
        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(cls)
        return point_set, cls


    def __len__(self):
        return len(self.filelist)

if __name__ == '__main__':

    datapath = os.path.join('..', '..', 'dataset')
    filelist_txt = os.path.join('..', '..', 'dataset', 'train.txt')

    d = ScanNet200Dataset(root = datapath, filelist_txt = filelist_txt, split = 'train')
    print(len(d))
    for batch in d:
        ps, cls = batch

        # print(ps.size(), ps.type(), cls.size(),cls.type())

        # print(cls)
    # d = ShapeNetDataset(root = datapath, classification = True)
    # print(len(d))
    # ps, cls = d[0]
    # print(ps.size(), ps.type(), cls.size(),cls.type())
    # get_segmentation_classes(datapath)

