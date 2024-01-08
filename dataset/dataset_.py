#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: dataset.py 
@time: 2020/06/30
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
from image_MEF_Fusion_deform.utils1.utils import *

class Dynamic_Scenes_Dataset(Dataset):
    def __init__(self, root_dir, is_training=True, crop=True, crop_size=None):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size
        self.scenes_dir_list = []
        for index in range(126,228):
            self.scenes_dir_list.append(index)
        self.scenes_dir_list_folder = ['trainA','trainB','trainC_']
        # scenes dir


             # self.scenes_dir = os.path.join(self.root_dir, 'Test/EXTRA')
        #     self.scenes_dir_list = os.listdir(self.scenes_dir)
        # else:
        #     self.scenes_dir = os.path.join(self.root_dir, 'Training')
        #     self.scenes_dir_list = os.listdir(self.scenes_dir)
        self.image_list = []
        for index in self.scenes_dir_list:
            self.folds = []
            for scene in range(len(self.scenes_dir_list_folder)):
                exp_pth1 = os.path.join(self.root_dir,self.scenes_dir_list_folder[scene],str(index)+'.png')
                self.folds.append(exp_pth1)
            self.image_list.append(self.folds)

    def __getitem__(self, index):
        # Read exposure times in one scene

        pre_img0 = Image.open(self.image_list[index][0]).convert("RGB")
        pre_img2 = Image.open(self.image_list[index][1]).convert("RGB")
        label = Image.open(self.image_list[index][2]).convert("RGB")
        pre_img0 = np.array(pre_img0)/255.0
        pre_img2 = np.array(pre_img2)/255.0
        label = np.array(label)/255.0
        # data argument
        if self.crop:
            H, W, _ = pre_img0.shape
            x = np.random.randint(0, H - self.crop_size[0] - 1)
            y = np.random.randint(0, W - self.crop_size[1] - 1)

            img0 = pre_img0[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
            # img1 = pre_img1[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
            label = label[x:x + self.crop_size[0], y:y + self.crop_size[1]].astype(np.float32).transpose(2, 0, 1)
        else:
            img0 = pre_img0.astype(np.float32).transpose(2, 0, 1)
            # img1 = pre_img1.astype(np.float32).transpose(2, 0, 1)
            img2 = pre_img2.astype(np.float32).transpose(2, 0, 1)
            label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        # print(np.shape(label))
        sample = {'input0': img0,  'input1': img2, 'label': label}
        return sample
    def __len__(self):
        return len(self.scenes_dir_list)