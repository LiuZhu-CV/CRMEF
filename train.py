#-*- coding:utf-8 _*-
"""
@author: LiuZhen
@license: Apache Licence
@file: train.py
@time: 2020/06/30
@contact: liuzhen.pwd@gmail.com
@site:
@software: PyCharm

"""
import argparse
import logging
import os
import sys
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import optim
from torchvision.models import vgg16
from tqdm import tqdm
from torch.utils.data import DataLoader

from PerceptualLoss import LossNetwork_per
from dataset.dataset import Dynamic_Scenes_Dataset
from pietorch.pytorch_ssim import ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from AHDRNAS import AHDR, AHDR_2, AHDR_4, AHDR_5, PixelDiscriminator, GANLoss, AHDR_MEF
from PerceptualLoss import new_loss_sobel as new_loss_sobel


def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_arch', type=int, default=1,
                        help='model architecture')
    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='training batch size (default: 2)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 2)')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--loss_func', type=int, default=0,
                        help='loss functions for training')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_decay_interval', type=int, default=2000,
                        help='decay learning rate every N epochs(default: 100)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--init_weights', type=bool, default=True,
                        help='init model weights')
    parser.add_argument('--logdir', type=str, default=None,
                        help='target log directory')
    parser.add_argument("--dataset_dir", type=str, default='/data/liuzhu/trainingdata/',
                        help='dataset directory')
    parser.add_argument('--validation', type=float, default=10.0,
                        help='percent of the data that is used as validation (0-100)')
    return parser.parse_args()

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def backward_D(real_B, fake_B, netD, criterionGAN):
    """Calculate GAN loss for the discriminator"""
    # Fake; stop backprop to the generator by detaching fake_B
    # fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = netD(fake_B)
    loss_D_fake = criterionGAN(pred_fake, False)
    # Real
    # real_AB = torch.cat((real_A, real_B), 1)
    pred_real = netD(real_B)
    loss_D_real = criterionGAN(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_D.backward()

def backward_G(fake_B,real_B,criterions,netD,criterionGAN):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = netD(fake_B)
        loss_G_GAN = criterionGAN(pred_fake, True)
        loss_total =  criterions[0](fake_B,real_B)*1.1 +(ssim(fake_B,real_B)*ssim_weight*(-1.0))*0.75
        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN*0.001 + loss_total
        loss_G.backward()

        return loss_G


def backward_G2(fake_, fake_B, real_B, criterions, netD, criterionGAN):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    # fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    pred_fake = netD(fake_B)
    loss_G_GAN = criterionGAN(pred_fake, True)
    loss_total =  criterions[0](fake_B, real_B) + criterions[1](fake_B, real_B) * 1.1 + (ssim(fake_B, real_B) * ssim_weight * (-1.0)) * 0.75
    # Second, G(A) = B
    # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
    # combine loss and calculate gradients
    loss_G = loss_G_GAN * 0.001 + loss_total
    loss_G.backward()

    return loss_G

ssim_weight = 0.8
def train(args, model, model_D, device, train_loader, optimizer, epoch, criterion,criterion2,optimizer_D,cri_D):
    model.train()
    epoch_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        batch_ldr0, batch_ldr1 = batch_data['input0'].to(device), batch_data['input1'].to(device)
        label = batch_data['label'].to(device)
        if batch_idx == 0:
            x1,pred = model(batch_ldr0, batch_ldr1)
        else:
            x1, pred = model(batch_ldr0_old, batch_ldr1_old)
        set_requires_grad(model_D,True)
        optimizer_D.zero_grad()
        backward_D(label,pred,model_D,cri_D)
        optimizer_D.step()
        set_requires_grad(model_D,False)
        # if batch_idx ==0:
        batch_ldr0_old = batch_ldr0
        batch_ldr1_old = batch_ldr1
        # else:

        x1,pred = model(batch_ldr0, batch_ldr1)
        optimizer.zero_grad()
        loss = backward_G2(x1,pred,label,[criterion,criterion2],model_D,cri_D)
        nn.utils.clip_grad_value_(model.parameters(), 0.01)
        optimizer.step()
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(batch_data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))
        if (epoch+1)%10 ==0:
            torchvision.utils.save_image(pred,'output1.png')
            torchvision.utils.save_image(batch_ldr0,'output2.png')
            torchvision.utils.save_image(batch_ldr1,'output3.png')
            torchvision.utils.save_image(label,'output4.png')
            torchvision.utils.save_image(x1, 'output2_1.png')
            torchvision.utils.save_image(x1, 'output2_2.png')


        if (epoch + 1) % 500 == 0:
            torch.save(model.state_dict(), './new-0521_AHDR4-'+str(epoch+1)+'.pt')



def main():
    # Settings
    args = get_args()

    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')


    # dataset and dataloader
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = Dynamic_Scenes_Dataset(root_dir='/data8T/sdx/Code/darts/train/', is_training=True,
                                  crop=True, crop_size=(160, 160))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    Genotype2 = namedtuple('Genotype2', 'c c1 m m1 f f1')
    genotype = Genotype2(c=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], c1=[1, 2, 3],
                         m=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], m1=[1, 2, 3],
                         f=[('resdilconv_3x3', 0), ('resdilconv_3x3', 1), ('resconv_3x3', 2), ('resconv_3x3', 3)], f1=
                         [1, 2, 3])

    # genotype2  = Genotype2(c=[('deformable_7', 0), ('deformable_5', 1), ('deformable_3', 2), ('deformable_5', 3)], c1=[1, 2, 3],
    #                                  m=[('conv_3x3', 0), ('conv_3x3', 1), ('conv_3x3', 2)], m1=[1, 2, 3],
    #                                  f=[('resdilconv_3x3', 0), ('resdilconv_3x3', 1), ('resdilconv_3x3', 2), ('resdilconv_3x3', 3)], f1=[1, 2, 3])
    genotype = eval("%s" % 'genotype')
    model = AHDR_MEF(64, genotype)
    model_D = PixelDiscriminator(3).to(device)
    
    # model.load_state_dict(torch.load('./new-0512_AHDR4-ct-3999.pt'))
    print('load sucessuful')
    # vgg_model = vgg16(pretrained=True).features[:16].cuda()
    # if args.init_weights:
    #     init_parameters(model)
    model.to(device)

    if args.loss_func == 0:
        criterion = nn.L1Loss()
    elif args.loss_func == 1:
        criterion = nn.MSELoss()
    else:
        print("Error loss functions.\n")
        return
    # criterion2 = LossNetwork_per(vgg_model).cuda()
    criterion2 = new_loss_sobel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr)

    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    print(f'''Starting training:
        Model Paras:     {num_parameters}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Loss function:   {args.loss_func}
        Learning rate:   {args.lr}
        Training size:   {len(train_loader)}
        Device:          {device.type}
        Dataset dir:     {args.dataset_dir}
        ''')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),eta_min=1e-10)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, float(args.epochs),eta_min=1e-6)
    cri_GAN = GANLoss('wgangp',)
    for epoch in range(1, args.epochs + 1):
        print('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train(args, model, model_D, device, train_loader, optimizer, epoch, criterion,criterion2,optimizer_D,cri_GAN)

        scheduler.step()
        scheduler_D.step()



if __name__ == '__main__':
    main()