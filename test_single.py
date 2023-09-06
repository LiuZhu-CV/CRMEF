
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import namedtuple

from AHDRNAS import AHDR_MEF
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
    genotype = Genotype2(c=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], c1=[1, 2, 3],
                         m=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], m1=[1, 2, 3],
                         f=[('resdilconv_3x3', 0), ('resdilconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3)], f1=
                         [1, 2, 3])

    # genotype2  = Genotype2(c=[('deformable_7', 0), ('deformable_5', 1), ('deformable_3', 2), ('deformable_5', 3)], c1=[1, 2, 3],
    #                                  m=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], m1=[1, 2, 3],
    #                                  f=[('resdilconv_3x3', 0), ('resdilconv_3x3', 1), ('resdilconv_3x3', 2), ('resdilconv_3x3', 3)], f1=[1, 2, 3])
    genotype = eval("%s" % 'genotype')
    model = AHDR_MEF(64, genotype).cuda()


    # genotype2  = Genotype2(c=[('deformable_7', 0), ('deformable_5', 1), ('deformable_3', 2), ('deformable_5', 3)], c1=[1, 2, 3],
    #                                  m=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], m1=[1, 2, 3],
    #                                  f=[('resdilconv_3x3', 0), ('resdilconv_3x3', 1), ('resdilconv_3x3', 2), ('resdilconv_3x3', 3)], f1=[1, 2, 3])
    # genotype = eval("%s" % 'genotype')
    # model = AHDR_4_light(6, 64, genotype).cuda()
    model.load_state_dict(torch.load('./new-0521_AHDR4-4000.pt', map_location='cuda:0'))
    # utils.load(model_2, './single_3DC/weights_2_1099.pt')
    # utils.load(model_1, './fusion0128/weights_1_1049.pt')
    # utils.load(model_2, './fusion0128/weights_2_1049.pt')
    model = model.eval()
    # model_2 = model_2.eval()

    # files_1 = os.listdir('./test/trainA/')
    # files_2 = os.listdir('./test/trainB/')
    files_1 = ['3.jpg','3.jpg','3.jpg','4.jpg','4.jpg','4.jpg']
    files_2 = ['5.jpg','6.jpg','7.jpg','5.jpg','6.jpg','7.jpg']
    print(len(files_1), len(files_2))

    index =0
    for name_1, name_2 in zip(files_1, files_2):
        index+=1
        file_name ='new_'+str(index)
        print(file_name)
        pth_out = './imgs_test/' + file_name + '.jpg'
        pth_out_f = './test/output_0808/' + file_name + '_fusion.jpg'
        pth_out_a1 = './test/output_0808/' + file_name + '_a1.jpg'
        pth_out_a2 = './test/output_0808/' + file_name + '_a2.jpg'
        pth_out_a3 = './test/output_0808/' + file_name + '_a3.jpg'
        pth_out_a4 = './test/output_0808/' + file_name + '_a4.jpg'
        pth_out_a5 = './test/output_0808/' + file_name + '_a5.jpg'

        print(name_1, name_2)
        # name_2 = name_1.split('low')[1]
        img1 = cv2.imread('./img/' + name_1)
        img2 = cv2.imread('./img/' + name_2)

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
            res,fused_1 = model(img1,img2)
            # f_ir, f_vis, fused = model_2(lr,vis, fused_1)
            # res = fused_1.data.cpu().numpy()
            # _,c, h, w= np.shape(res)
            torchvision.utils.save_image(fused_1,pth_out)
            # torchvision.utils.save_image(res,pth_out_f)
            # torchvision.utils.save_image(att_list[0],pth_out_a1)
            # torchvision.utils.save_image(att_list[1], pth_out_a2)
            # torchvision.utils.save_image(att_list[2], pth_out_a3)
            # torchvision.utils.save_image(att_list[3], pth_out_a4)
            # torchvision.utils.save_image(att_list[4], pth_out_a5)

    print('Test done.')
