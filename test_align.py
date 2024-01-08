# python 2.7, pytorch 0.3.1

import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import namedtuple

from AHDRNAS import  AHDR_5
import torch
import cv2
import shutil
import torchvision
import numpy as np
import itertools
import subprocess
import random

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ski_ssim
# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()

if __name__ == '__main__':

    Genotype2 = namedtuple('Genotype2', 'c c1 m m1 f f1')
    Genotype2 = namedtuple('Genotype2', 'c c1 m m1 f f1')
    genotype = Genotype2(c=[('deformable_5', 0), ('deformable_5', 1), ('deformable_5', 2), ('deformable_3', 3)],
                         c1=[1, 2, 3], m=[('conv_3x3', 0), ('conv_3x3', 1), ('dilconv_3x3', 2)], m1=[1, 2, 3],
                         f=[('resconv_3x3', 0), ('resdilconv_3x3', 1), ('resdilconv_3x3', 2), ('resconv_3x3', 3)],
                         f1=[1, 2, 3])

    # genotype2  = Genotype2(c=[('deformable_7', 0), ('deformable_5', 1), ('deformable_3', 2), ('deformable_5', 3)], c1=[1, 2, 3],
    #                                  m=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], m1=[1, 2, 3],
    #                                  f=[('resdilconv_3x3', 0), ('resdilconv_3x3', 1), ('resdilconv_3x3', 2), ('resdilconv_3x3', 3)], f1=[1, 2, 3])
    genotype = eval("%s" % 'genotype')
    model = AHDR_5(6, 64, genotype).cuda()
    model.load_state_dict(torch.load('./new-0515_AHDR-align-ct-3999.pt', map_location='cuda:0'))
    # utils.load(model_2, './single_3DC/weights_2_1099.pt')
    # utils.load(model_1, './fusion0128/weights_1_1049.pt')
    # utils.load(model_2, './fusion0128/weights_2_1049.pt')
    model = model.eval()
    # model_2 = model_2.eval()

    files_1 = os.listdir('./test_align/trainA/')
    files_2 = os.listdir('./test_align/trainB/')

    print(len(files_1), len(files_2))
    for name_1, name_2 in zip(files_1, files_2):
        file_name = name_1.split('.')[0]
        print(file_name)
        pth_out = './test_align/' + file_name + '.jpg'
        print(name_1, name_2)

        img1 = cv2.imread('./test_align/trainA/' + name_1)
        img2 = cv2.imread('./test_align/trainB/' +name_1)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img1 = np.array(img1)[np.newaxis, :]
        img1 = np.transpose(img1, (0, 3, 1, 2)).astype(np.float) / 255.
        # print(np.shape(im_input), im_input)
        img1 = torch.tensor(img1).type(torch.FloatTensor)
        img1 = Variable(img1, requires_grad=False).cuda()

        img2 = np.array(img2)[np.newaxis, :]
        img2 = np.transpose(img2, (0, 3, 1, 2)).astype(np.float) / 255.
        # print(np.shape(im_input), im_input)
        img2 = torch.tensor(img2).type(torch.FloatTensor)
        img2 = Variable(img2, requires_grad=False).cuda()
        with torch.no_grad():
            _,_,fused_1 = model(img1,img2)
            # f_ir, f_vis, fused = model_2(lr,vis, fused_1)
            res = fused_1.data.cpu().numpy()
            _,c, h, w= np.shape(res)
            torchvision.utils.save_image(fused_1,pth_out)
    print('Test done.')
