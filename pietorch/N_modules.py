import numpy as np
import torch
import random
import torchvision
import math
import torch.nn as nn
import itertools
import skimage as ski
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from scipy.special import gamma
from skimage.transform import warp
import cv2


class InsNorm(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(InsNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        flat_len = x.size(2) * x.size(3)
        vec = x.view(x.size(0), x.size(1), flat_len)
        mean = torch.mean(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((flat_len - 1) / float(flat_len))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class SELayer(nn.Module):
    def __init__(self, channel, reduction=64):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def CropSample(im_input, label, crop_size):
    if isinstance(label, np.ndarray):
        label = Image.fromarray(label)
    if isinstance(im_input, np.ndarray):
        im_input = Image.fromarray(im_input)

    W, H = label.size
    x_offset = random.randint(0, W - crop_size)
    y_offset = random.randint(0, H - crop_size)
    label = label.crop((x_offset, y_offset,
                        x_offset + crop_size, y_offset + crop_size))
    im_input = im_input.crop((x_offset, y_offset,
                              x_offset + crop_size, y_offset + crop_size))
    return im_input, label


def CropSample3(im_input, label, trans, atms, streaks, size):
    output_list = []
    input_list = [im_input, label, trans, atms, streaks]
    num_of_length = len(input_list)
    h, w, c = input_list[0].shape
    try:
        row = np.random.randint(h - size)
        col = np.random.randint(w - size)
    except:
        print("random low value leq high value")
        print(h, w, c)
    for i in range(num_of_length - 1):
        item = input_list[i]
        item = item[row:row + size, col:col + size, :]
        output_list.append(item)
    h, w, c = input_list[-1].shape
    row = np.random.randint(h - size)
    col = np.random.randint(w - size)
    item = input_list[-1][row:row + size, col:col + size, :]
    output_list.append(item)
    assert (len(input_list) == len(output_list))
    return output_list[0],output_list[1],output_list[2],output_list[3],output_list[4]


def CropSample2(im_input, crop_size):
    if isinstance(im_input, np.ndarray):
        im_input = Image.fromarray(im_input)

    W, H = im_input.size
    x_offset = random.randint(0, W - crop_size)
    y_offset = random.randint(0, H - crop_size)

    im_input = im_input.crop((x_offset, y_offset,
                              x_offset + crop_size, y_offset + crop_size))
    return im_input


def DataAugmentation(im_input, label):
    if random.random() > 0.5:
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        im_input = im_input.transpose(Image.FLIP_LEFT_RIGHT)
    #    if random.random() > 0.5:
    #        label    = label.transpose(   Image.FLIP_TOP_BOTTOM)
    #        im_input = im_input.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        label = label.rotate(angle)
        im_input = im_input.rotate(angle)
    return im_input, label

def DataAugmentation3(im_input, label,trans, atms, streaks):
    output_list = []
    input_list = [im_input, label, trans, atms, streaks]
    if np.random.rand() < 0.5:
        for item in input_list:
            output_list.append(np.copy(np.fliplr(item)))
    else:
        output_list = input_list
    return output_list[0],output_list[1],output_list[2],output_list[3],output_list[4]
