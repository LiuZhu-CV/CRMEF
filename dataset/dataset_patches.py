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
from torch.utils.data import Dataset
import os
import random
rotation = True
import pickle
def get_patch_from_file(pkl_path, pkl_id):
    with open(pkl_path + '/' + str(pkl_id) + '.pkl', 'rb') as pkl_file:
        res = pickle.load(pkl_file)
    return res

def data_augmentation(image, mode):
    '''
    Performs dat augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        pass
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out

class Dynamic_Scenes_Dataset(Dataset):
    def __init__(self, patch_dir, is_training=True, crop=True, crop_size=None):
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size

        # scenes dir
        self.patch_path = patch_dir
        self.count = len(os.listdir(self.patch_path))


    def __getitem__(self, index):
        if self.crop:

            data = get_patch_from_file(self.patch_path, index+1)
            img0 = data['in_LDR_1']
            img1 = data['in_LDR_2']
            label = data['ref_HDR']

            if rotation and self.crop:
                flag_aug = random.randint(1, 7)
                img1 =data_augmentation(img1,flag_aug)
                img0 = data_augmentation(img0, flag_aug)
                label = data_augmentation(label, flag_aug)
            img0 = img0.transpose(2, 0, 1)
            img1 = img1.transpose(2, 0, 1)
            label = label.transpose(2, 0, 1)
            img0 = torch.from_numpy(img0.copy())
            img1 = torch.from_numpy(img1.copy())
            label = torch.from_numpy(label.copy())

            sample = {'input0': img0, 'input1': img1, 'label': label}

        return sample

    def __len__(self):
        return self.count