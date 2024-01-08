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
from skimage import transform

from .N_modules import CropSample, DataAugmentation, CropSample2, CropSample3, DataAugmentation3
import cv2
def resize_get(w):
    if w%4==0:
        return int(w)
    if (w-1)%4==0:
        return int(w-1)
    if (w-2)%4==0:
        return int(w-2)
    if (w+1)%4==0:
        return int(w+1)
    if (w+2)%4==0:
        return int(w+2)
class ConvertImageSet(data.Dataset):
    # Init. it.
    def __init__(self, dataroot,
                 imlist_pth,
                 data_name,
                 transform=None,
                 resize_to=None,
                 crop_size=None,
                 is_train =False,
                 with_aug =False):
        self.is_train  = is_train
        self.with_aug  = with_aug
        self.dataroot  = dataroot
        self.transform = transform
        self.data_name = data_name
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.imlist    = self.flist_reader(imlist_pth)        
        self.count = 0
    # Process data.
    def __getitem__(self, index):
        im_name = self.imlist[index]
        if self.data_name != 'hazerain2':
            im_input, label = self.sample_loader(im_name)

            # print(self.count,'count 448')
            # Resize a sample, or not.
            # im_input = im_input.crop((1, 1, 129, 129))
            # label = label.crop((1, 1, 129, 129))
            if not self.resize_to is None:
                h = np.shape(np.array(im_input))[0]
                w = np.shape(np.array(im_input))[1]
                # print(h,w)
                # w = 320
                # h = 240
                # h = 160
                # w = 240
                if (h%4==0) and (w%4==0):
                    self.resize_to =(w,h)
                else:
                    self.resize_to = (resize_get(w),resize_get(h))
                    print(self.resize_to)
                w =self.resize_to[0]
                h =self.resize_to[1]
                if (h==448 and w ==640) :
                    # h=384
                    # w=512
                    self.resize_to =(w,h)
                    self.count+=1

                if (h==1456 and w ==2592) or(h==2212 and w ==3908):
                    print(im_name)
                    h=448
                    w=640
                    self.resize_to =(w,h)
                    im_input = cv2.resize(np.array(im_input), self.resize_to)
                    label = cv2.resize(np.array(label), self.resize_to)
                print(np.shape(im_input))
                if w == 480 and h ==320:
                    im_input = im_input.crop((1,1,481,321))
                    label = label.crop((1,1,481,321))
                elif w == 320 and h ==480:
                    im_input = im_input.crop((1,1,321,481))
                    label = label.crop((1,1,321,481))
                elif w !=512 or h!= 384:
                    im_input = cv2.resize(np.array(im_input), self.resize_to)
                    label = cv2.resize(np.array(label), self.resize_to)


            if not (self.transform is None):
                im_input, label = self.Transformer(im_input, label)
        else:
            im_input, label, trans, atms, streaks = self.sample_loader(im_name)
            # if not self.resize_to is None:
            #     w = self.resize_to[0]
            #     h = self.resize_to[1]
            #     im_input = cv2.resize(np.array(im_input), self.resize_to)
            #     label = cv2.resize(np.array(label), self.resize_to)
            if not (self.transform is None):
                im_input, label, trans, atms, streaks= self.Transformer2(im_input, label,trans,atms,streaks)
                return im_input,label,trans, atms, streaks, im_name
        return im_input, label, im_name


    # Read a image name list.
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist


    # Return a pair of images (input, label).
    def sample_loader(self, im_name):
        if self.data_name   == 'RESIDE':
            return RESIDE_loader(self.dataroot, im_name, self.is_train)

        elif self.data_name == 'DCPDNData':
            return DCPDNData_loader(self.dataroot, im_name)

        elif self.data_name == 'BSD_gray':
            return BSDgray_loader(self.dataroot, im_name)
        elif self.data_name == 'BSD68':
            return BSD68_loader(self.dataroot, im_name)
        elif self.data_name == 'Set12':
            return Set12_loader(self.dataroot, im_name)
        elif self.data_name == 'RealNoiseHKPoly':
            return RealNoiseHKPoly_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'SIDD':
            return SIDD_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'nam':
            return nam_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'nus':
            return nus_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'SIDDnew':
            return SIDDnew_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'nora':
            return nora_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'test1':
            return test1_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'LOL':
            return LOL_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'RealSR':
            return RealSR_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'DND':
            return DND_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'GoPro':
            return GoPro_loader(self.dataroot, im_name)

        elif self.data_name == 'CarDataset':
            return Car_loader(self.dataroot, im_name)

        elif self.data_name == 'RainDrop':
            return RainDrop_loader(self.dataroot, im_name, self.is_train, color_fmt='BGR')

        elif self.data_name == 'DDN_Data':
            return DDNdata_loader(self.dataroot, im_name, self.is_train)

        elif self.data_name == 'DIDMDN_Data':
            return DIDMDNdata_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'rain800':
            return rain800_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'rain800c':
            return rain800_classify_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'nonblind':
            return nonblind_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'rain100h':
            return rain100h_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'rain100l':
            return rain100l_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'textblur':
            return Textblur_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'PET':
            return PET_loader(self.dataroot,im_name,self.is_train)
        elif self.data_name == 'hazerain':
            return Hazerain_loader(self.dataroot,im_name,self.is_train)
        elif self.data_name == 'hazerain2':
            return Hazerain_loader2(self.dataroot, im_name, self.is_train)
        elif self.data_name == 'fusion':
            return fusion_loader(self.dataroot, im_name, self.is_train)
        elif self.data_name =='me':
            return me_loader(self.dataroot,im_name,self.is_train)
        elif self.data_name =='mf':
            return mf_loader(self.dataroot,im_name,self.is_train)
        else:
            print("Unknown dataset.")
            quit()
    # def Transformer2(self,im_input,label,kernel):
    #     if self.data_name == 'nonblind':
    #         im_input = (self.transform(im_input)[0,:,:].unsqueeze(0))
    #         label = (self.transform(label)[0,:,:].unsqueeze(0))
    #         kernel = torch.FloatTensor(kernel)
    #         return im_input,label,kernel
    def Transformer2(self, im_input, label, trans, atms, streaks):
        if self.data_name == 'hazerain2':
            if not self.crop_size is None:
                im_input, label, trans, atms, streaks = CropSample3(im_input, label, trans, atms,streaks, self.crop_size)
            if self.with_aug:
                im_input, label,trans,atms,streaks = DataAugmentation3(im_input, label, trans, atms, streaks)
            im_input = self.transform(im_input)
            label    = self.transform(label)
            trans = self.transform(trans)
            atms = self.transform(atms)
            streaks = self.transform(streaks)
            return im_input, label, trans, atms, streaks
    def Transformer(self, im_input, label):
        if self.data_name == 'RESIDE':
            if not self.is_train:
                label = self.transform(label)
                im_input = im_input.transpose((3, 2, 0, 1))
                im_input = torch.FloatTensor(im_input)
                im_input/= 255.0
            else:
                if not self.crop_size is None:
                    im_input, label = CropSample(im_input, label, self.crop_size)
                if self.with_aug:
                    im_input, label = DataAugmentation(im_input, label)

                im_input = self.transform(im_input)
                label    = self.transform(label)

        elif self.data_name == 'DCPDNData':
            im_input = im_input.transpose((2, 0, 1))
            im_input = torch.FloatTensor(im_input)
            label = label.transpose((2, 0, 1))
            label = torch.FloatTensor(label)

        elif self.data_name in ['RainDrop',
                                'GoPro',
                                'CarDataset',
                                'RealNoiseHKPoly',
                                'DIDMDN_Data','DIDMDN_Datal','DIDMDN_Datam','DIDMDN_Datah','rain800','rain100h','rain100l',
                                'textblur','rain800c','SIDD','nora','SIDDnew','nus','nam','test1','DND','RealSR','LOL','hazerain','fusion','me','mf']:
            if not self.crop_size is None and self.data_name !='rain800c':
                im_input, label = CropSample(im_input, label, self.crop_size)
            elif not self.crop_size is None:
                # print('crop')
                im_input= CropSample2(im_input, self.crop_size)
            if self.with_aug:
                im_input, label = DataAugmentation(im_input, label)
            if self.data_name =='PET' or self.data_name =='rain800c':
                im_input = self.transform(im_input)
                # print(label)
                # label    = torch.Tensor(l)
                # print(np.shape(im_input),np.shape(label))
            else:
                im_input = self.transform(im_input)
                label    = self.transform(label)
            if self.data_name == 'textblur':
                im_input = AddGaussianNoise(im_input, 1)
        elif self.data_name == 'BSD_gray' or self.data_name == 'Set12' or self.data_name=='BSD68':
            if not self.is_train:
                transf, noise_level = self.transform
                im_input = transf(im_input)
                im_input = AddGaussianNoise(im_input, noise_level)
                im_input =  im_input[0,:,:].unsqueeze(0)
                label = transf(label)
                label = label[0,:,:].unsqueeze(0)
            else:
                label = self.transform(label)
                im_input = label.clone()

        elif self.data_name == 'DDN_Data':
            label = self.transform(label)
            im_input = im_input.transpose((3, 2, 0, 1))
            im_input = torch.FloatTensor(im_input)
            im_input/= 255.0
        else:
            pass

        return im_input, label

    def __len__(self):
        return len(self.imlist)

def Textblur_loader(dataroot, im_name, is_train):
    label_pth = dataroot+im_name+'orig.png'
    label = Image.open(label_pth).convert("RGB")
    label = label.resize((256, 256))
    blur_pth = dataroot + im_name + 'blur.png'
    blur = Image.open(blur_pth).convert("RGB")
    blur = blur.resize((256, 256))
    return blur, label


def RESIDE_loader(dataroot, im_name, is_train):
    if not is_train:
        Vars = np.arange(1, 11, 1)
        label_pth = dataroot+'labels/'+im_name
        label = Image.open(label_pth).convert("RGB")
        # label = label.resize((512,384))
        for var in Vars:
            if var == 1:
                hazy = np.asarray(Image.open(
                    dataroot+'images/'+im_name.split('.')[0]+'_'+str(var)+'.png'))
                hazy = Image.fromarray(hazy)
                # hazy = hazy.resize((512,384))

                hazy = np.expand_dims(np.asarray(hazy), axis=3)
            else:
                current = np.asarray(Image.open(
                    dataroot+'images/'+im_name.split('.')[0]+'_'+str(var)+'.png'))
                current = Image.fromarray(current)
                # current = current.resize((512,384))
                current = np.expand_dims(np.asarray(current), axis=3)
                hazy = np.concatenate((hazy, current), axis=3)
    else:
        var = random.choice(np.arange(1, 11, 1))
        label_pth = dataroot+'labels/'+im_name
        hazy_pth = dataroot+'images/'+im_name.split('.')[0]+'_'+str(var)+'.png'

        label = Image.open(label_pth).convert("RGB")
        hazy  = Image.open(hazy_pth).convert("RGB")

    return hazy, label


def DCPDNData_loader(dataroot, im_name):
    sample_pth = dataroot+im_name
    f = h5py.File(sample_pth, 'r')
    keys = f.keys()

    # h5 to numpy, ato and trm are not used.
#    ato = np.asarray(f[keys[0]])
    label = np.asarray(f[keys[1]])
    hazy = np.asarray(f[keys[2]])
#    trm = np.asarray(f[keys[3]])

    if label.max() > 1 or hazy.max() > 1:
        print("DCPDNData out of range [0, 1].")
        quit()
    return hazy, label


def RainDrop_loader(dataroot, im_name, is_train, color_fmt='BGR'):
    if not is_train:
        if dataroot.split('/')[-2] == 'test_a':
            houzhui = 'png'
        else:
            assert(dataroot.split('/')[-2] == 'test_b')
            houzhui = 'jpg'

        label = cv2.imread(dataroot+'gt/'+im_name+'_clean.'+houzhui)
        label = align_to_k(label, k=4)
        rainy = cv2.imread(dataroot+'data/'+im_name+'_rain.'+houzhui)
        rainy = align_to_k(rainy, k=4)

        if color_fmt == 'RGB':
            rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2RGB)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    else:
        label = cv2.imread(dataroot+'gt/'  +im_name+'_clean.png')
        rainy = cv2.imread(dataroot+'data/'+im_name+'_rain.png')

    return rainy, label

def GoPro_loader(dataroot, im_name):
    name1, name2 = im_name.split('/')
    blur_pth  = dataroot+name1+'/blur/'+name2
    label_pth = dataroot+name1+'/sharp/'+name2
    blur = cv2.imread(blur_pth)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    label = cv2.imread(label_pth)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    return blur, label


def PET_loader(dataroot, im_name, is_train =False):
    if is_train:
        label = 1
        var = im_name.split('_')[0]
        img = im_name.split('_')[1]
        if var =='CN':
            label = 0
        image = dataroot+var+'/'+img
        image = Image.open(image).convert('RGB')
        angle = random.choice([45,90,135,180,225,270,315,360])
        image = transform.rotate(np.array(image), angle)
        image = Image.fromarray(np.array(image*255).astype(np.uint8))
    else:
        image = dataroot  + im_name
        image = Image.open(image).convert('RGB')
        label = 0
    return image,label

def Car_loader(dataroot, im_name):
    blur_pth  = dataroot+'/blurred/'+im_name
    label_pth = dataroot+'/sharp/'+im_name

    blur = cv2.imread(blur_pth)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    label = cv2.imread(label_pth)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    return blur, label


def BSDgray_loader(dataroot, im_name):
    im_pth = dataroot+im_name
    label  = Image.open(im_pth).convert('RGB')
    noisy  = Image.open(im_pth).convert('RGB') # to which noise will be added (in test).
    return noisy, label

def BSD68_loader(dataroot, im_name):
    im_pth = dataroot+im_name
    label  = Image.open(im_pth).convert('RGB')
    noisy  = Image.open(im_pth).convert('RGB') # to which noise will be added (in test).
    return noisy, label
def Set12_loader(dataroot, im_name):
    im_pth = dataroot+im_name
    label  = Image.open(im_pth).convert('RGB')
    noisy  = Image.open(im_pth).convert('RGB') # to which noise will be added (in test).
    return noisy, label
def RealNoiseHKPoly_loader(dataroot, im_name, is_train):
    if is_train:
        noisy_pth = dataroot+im_name.split('mean')[0]+'Real.JPG'
    else:
        noisy_pth = dataroot+im_name.split('mean')[0]+'real.PNG'
    label_pth = dataroot+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label

def SIDD_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        pth,number,index = im_name.split('|')
        noisy_pth = dataroot+pth+'/'+number+'_NOISY_SRGB_'+index+'.PNG'
        label_pth = dataroot+pth+'/'+number+'_GT_SRGB_'+index+'.PNG'
    else:
        noisy_pth = dataroot+'input/'+im_name
        label_pth = dataroot+'groundtruth/'+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label

def me_loader(dataroot, im_name, istrain):
    if istrain:
        # noisy_pth = dataroot + '/o/' + im_name + '.png'
        # label_pth = dataroot + '/u/' + im_name + '.png'
        noisy_pth = dataroot + '/JPEGImages_ir/' + im_name + '.png'
        label_pth = dataroot + '/JPEGImages_vis/' + im_name + '.png'
        noisy = Image.open(noisy_pth)
        label = Image.open(label_pth).convert('L')
        return noisy, label


def mf_loader(dataroot, im_name, istrain):
    if istrain:
        noisy_pth = dataroot + '/f/' + im_name + '.png'
        label_pth = dataroot + '/n/' + im_name + '.png'
        noisy = Image.open(noisy_pth)
        label = Image.open(label_pth)
        return noisy, label



def fusion_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        pth1, pth2 = im_name.split('>>')
        noisy_pth = dataroot+pth1
        label_pth = dataroot+pth2
    else:
        pth1, pth2 = im_name.split('>>')
        noisy_pth = dataroot + pth1
        label_pth = dataroot + pth2
    noisy = Image.open(noisy_pth)
    label = Image.open(label_pth)
    # print(np.shape(np.array(noisy)))
    # print(np.shape(np.array(label)))
    return noisy, label

def nam_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        noisy_pth = dataroot + 'Nam_patches/' + im_name + '_noise.png'
        label_pth = dataroot + 'Nam_patch_GT/' + im_name + '_gt.png'
    else:
        noisy_pth = dataroot+'Nam_patches/'+im_name+'_noise.png'
        label_pth = dataroot+'Nam_patch_GT/'+im_name+'_gt.png'
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label
def test1_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        noisy_pth = dataroot + 'in/' + im_name + '_s80_a04.png'
        label_pth = dataroot + 'gt/' + im_name + '.png'
    else:
        noisy_pth = dataroot+'in/'+im_name+'_s80_a04.png'
        label_pth = dataroot+'gt/'+im_name+'.png'
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label
def nus_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        noisy_pth = dataroot + 'in/' + im_name + '_s80_a04.png'
        label_pth = dataroot + 'gt/' + im_name + '.png'
    else:
        noisy_pth = dataroot+'/'+im_name
        label_pth = dataroot+'/'+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label
def SIDDnew_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        # pth,number,index = im_name.split('|')
        noisy_pth = dataroot+'/train/noisy_'+im_name+'.png'
        label_pth = dataroot+'/train/label_'+im_name+'.png'
    else:
        noisy_pth = dataroot+'input/'+im_name
        label_pth = dataroot+'groundtruth/'+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label
def nora_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        # pth,number,index = im_name.split('|')
        noisy_pth = dataroot+'/nl_'+im_name+'.png'
        label_pth = dataroot+'/nr_'+im_name+'.png'
    else:
        noisy_pth = dataroot+'/nl_'+im_name
        label_pth = dataroot+'/nr_'+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return  label,noisy
def LOL_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        noisy_pth = dataroot + 'low/' + im_name
        label_pth = dataroot + 'high/' + im_name
    else:
        noisy_pth = dataroot+'low/'+im_name
        label_pth = dataroot+'high/'+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label
def RealSR_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        pth,number,index = im_name.split('|')
        noisy_pth = dataroot+pth+'/'+number+'_NOISY_SRGB_'+index+'.PNG'
        label_pth = dataroot+pth+'/'+number+'_GT_SRGB_'+index+'.PNG'
    else:
        noisy_pth = dataroot+'LR/'+im_name
        label_pth = dataroot+'HR/'+im_name

    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')

    return noisy, label
def DND_loader(dataroot, im_name, is_train):
    #GT_SRGB,NOISY_SRGB
    if is_train:
        pth,number,index = im_name.split('|')
        noisy_pth = dataroot+pth+'/'+number+'_NOISY_SRGB_'+index+'.PNG'
        label_pth = dataroot+pth+'/'+number+'_GT_SRGB_'+index+'.PNG'
    else:
        noisy_pth = dataroot+'input/'+im_name
        label_pth = dataroot+'input/'+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label


def rain100h_loader(dataroot, im_name, is_train):
    if is_train:
        noisy_pth = dataroot + 'norain-' + im_name+'.png'
        label_pth = dataroot + 'rain-' + im_name+'.png'
        noisy = Image.open(noisy_pth).convert('RGB')
        label = Image.open(label_pth).convert('RGB')
    else:
        noisy_pth = dataroot + '/norain/norain-' + im_name + '.png'
        label_pth = dataroot + '/rain/X2/norain-' + im_name + 'x2.png'
        # noisy_pth = dataroot + '/' + im_name + '.png'
        # label_pth = dataroot + '/' + im_name + '.png'
        noisy = Image.open(noisy_pth).convert('RGB')
        label = Image.open(label_pth).convert('RGB')
    return label, noisy

def rain100l_loader(dataroot, im_name, is_train):
    if is_train:
        noisy_pth = dataroot + 'norain/norain-' + im_name+'.png'
        label_pth = dataroot + 'rain/norain-' + im_name+'x2.png'
        noisy = Image.open(noisy_pth).convert('RGB')
        label = Image.open(label_pth).convert('RGB')
    else:
        noisy_pth = dataroot + 'norain/norain-' + im_name + '.png'
        label_pth = dataroot + 'rain/X2/norain-' + im_name + 'x2.png'
        # noisy_pth = dataroot + '/' + im_name + '.png'
        # label_pth = dataroot + '/' + im_name + '.png'
        noisy = Image.open(noisy_pth).convert('RGB')
        label = Image.open(label_pth).convert('RGB')
    return label,noisy

def Hazerain_loader(dataroot, im_name, is_train):

    if is_train:
        name_list = im_name.split('_')
        name = name_list[0]
        index = name_list[1]
        noisy_pth = dataroot + 'in/' +im_name
        label_pth = dataroot + 'gt/' +name+'_'+index+'.png'
        noisy = Image.open(noisy_pth).convert('RGB')
        label = Image.open(label_pth).convert('RGB')
    else:
        name_list = im_name.split('_')
        name = name_list[0]
        index = name_list[1]
        noisy_pth = dataroot + 'in/' + name + '_' + index + '.png'
        label_pth = dataroot + 'gt/' + name + '_' + index + '.png'
        noisy = Image.open(noisy_pth).convert('RGB')
        label = Image.open(label_pth).convert('RGB')
    return noisy,label

def Hazerain_loader2(dataroot, im_name, is_train):
    name_list = im_name.split('_')
    name = name_list[0]
    index = name_list[1]
    noisy_pth = dataroot + 'in/' +im_name
    # trans,atm and rain streaks
    label_pth = dataroot + 'gt/' +name+'_'+index+'.png'
    tran_pth = dataroot+'trans/'+im_name
    atm_pth = dataroot+'atm/'+im_name
    streak_pth = dataroot+'streak/'+im_name
    noisy = read_image(noisy_pth)
    label = read_image(label_pth)
    # get this and transform this
    trans = read_image(tran_pth)
    atms = read_image(atm_pth)
    streak = read_image(streak_pth)
    return noisy,label, trans,atms,streak

# read image by Liruoteng
def read_image(image_path, noise=False):
    """
    function: read image function
    :param image_path: input image path
    :param noise: whether apply noise on image
    :return: image in numpy array, range [0,1]
    """
    img_file = Image.open(image_path)
    img_data = np.array(img_file, dtype=np.float32)
    (h,w,c) = img_data.shape
    if len(img_data.shape) < 3:
        img_data = np.dstack((img_data, img_data, img_data))
    if noise:
        (h,w,c) = img_data.shape
        noise = np.random.normal(0,1,[h,w])
        noise = np.dstack((noise, noise, noise))
        img_data = img_data + noise
    img_data = img_data.astype(np.float32)/255.0
    img_data[img_data > 1.0] = 1.0
    img_data[img_data < 0] = 0.0
    return img_data.astype(np.float32)


def rain800_loader(dataroot, im_name, is_train):
    pair_pth = dataroot+im_name
    pair = Image.open(pair_pth)
    # pair = pair.resize((160,160))
    pair_w, pair_h = pair.size
    label = pair.crop((0, 0, pair_w/2, pair_h))
    rainy_ = gasuss_noise(label)
    rainy = Image.fromarray(rainy_)
    # rainy = pair.crop((pair_w/2, 0, pair_w, pair_h))
    return rainy, label
def nonblind_loader(dataroot, im_name, is_train):
    # print(im_name)
    strs = im_name.split('_')
    index = strs[1]
    kid = strs[2]
    # the solutions of the model\
    n_pth = dataroot+'x_'+str(int(index))+'.png'
    r_pth = dataroot+'y_'+str(int(index))+'_'+kid+'.png'

    k_pth = dataroot+'levin_data/kernel_'+str(int(kid))+'.dlm'
    noisy = Image.open(n_pth).convert('RGB')
    label = Image.open(r_pth).convert('RGB')
    ker = np.loadtxt(k_pth)[np.newaxis,...].astype(np.float64)
    ker = np.clip(ker, 0, 1)
    ker = ker / np.sum(ker)

    return label,noisy,ker
def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(np.array(image)/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def rain800_classify_loader(dataroot, im_name, is_train):
    pair_pth = dataroot+im_name
    pair = Image.open(pair_pth)
    # pair = pair.resize((160,160))
    pair_w, pair_h = pair.size
    label = pair.crop((0, 0, pair_w/2, pair_h))
    # rainy = pair.crop((pair_w/2, 0, pair_w, pair_h))
    rainy_ = gasuss_noise(label)
    rainy = Image.fromarray(rainy_)
    newlabel = random.choice([0,1])
    if newlabel ==0:
        image = label
    else:
        image = rainy
    return image, newlabel

def DIDMDNdata_loader(dataroot, im_name, is_train):
    if is_train:
        var      = random.choice(['Rain_Heavy', 'Rain_Medium', 'Rain_Light'])
        pair_pth = dataroot+var+'/train2018new/'+im_name
    else:
        pair_pth = dataroot+im_name

    pair = Image.open(pair_pth)
    pair_w, pair_h = pair.size
    rainy = pair.crop((0, 0, pair_w/2, pair_h))
    label = pair.crop((pair_w/2, 0, pair_w, pair_h))
    return rainy, label    

def DIDMDNdata_loaderl(dataroot, im_name, is_train):
    if is_train:
        var = 'Rain_Light'
        pair_pth = dataroot+var+'/train2018new/'+im_name
    else:
        pair_pth = dataroot+im_name

    pair = Image.open(pair_pth)
    pair_w, pair_h = pair.size
    rainy = pair.crop((0, 0, pair_w/2, pair_h))
    label = pair.crop((pair_w/2, 0, pair_w, pair_h))
    return rainy, label

def DIDMDNdata_loaderm(dataroot, im_name, is_train):
    if is_train:
        var      = 'Rain_Medium'
        pair_pth = dataroot+var+'/train2018new/'+im_name
    else:
        pair_pth = dataroot+im_name

    pair = Image.open(pair_pth)
    pair_w, pair_h = pair.size
    rainy = pair.crop((0, 0, pair_w/2, pair_h))
    label = pair.crop((pair_w/2, 0, pair_w, pair_h))
    return rainy, label

def DIDMDNdata_loaderh(dataroot, im_name, is_train):
    if is_train:
        var      = 'Rain_Heavy'
        pair_pth = dataroot+var+'/train2018new/'+im_name
    else:
        pair_pth = dataroot+im_name

    pair = Image.open(pair_pth)
    pair_w, pair_h = pair.size
    rainy = pair.crop((0, 0, pair_w/2, pair_h))
    label = pair.crop((pair_w/2, 0, pair_w, pair_h))
    return rainy, label
    
def DDNdata_loader(dataroot, im_name, is_train):    
    label_pth = dataroot+'label/'+im_name
    label = Image.open(label_pth).convert("RGB")
    
    if is_train:
        var = random.choice(np.arange(1, 15, 1))
        rainy_pth = dataroot+'rain_image/'+im_name.split('.')[0]+str(var)+'.jpg'
        rainy = Image.open(rainy_pth).convert("RGB")
    else:
        for var in np.arange(1, 15, 1):
            if var == 1:
                rainy = np.asarray(Image.open(
                    dataroot+'rain_image/'+im_name.split('.')[0]+'_'+str(var)+'.jpg'))            
                rainy = np.expand_dims(rainy, axis=3)
                
            else:
                current = np.asarray(Image.open(
                    dataroot+'rain_image/'+im_name.split('.')[0]+'_'+str(var)+'.jpg'))            
                current = np.expand_dims(current, axis=3) 
                rainy   = np.concatenate((rainy, current), axis=3)
    
    return rainy, label
   
    
def align_to_k(img, k=4):
    a_row = int(img.shape[0]/k)*k
    a_col = int(img.shape[1]/k)*k
    img = img[0:a_row, 0:a_col]
    return img


def AddGaussianNoise(patchs, var):
    # A randomly generated seed. Use it for an easy performance comparison.
    # m_seed_cpu = 8526081014239199321
    # m_seed_gpu = 8223752412272754
    # torch.cuda.manual_seed(m_seed_gpu)
    # torch.manual_seed(m_seed_cpu)
    patchs = np.array(patchs)
    c, h, w = patchs.size()
    noise_pad = torch.FloatTensor(c, h, w).normal_(0, var)
    noise_pad = torch.div(noise_pad, 255.0)
    patchs+= noise_pad
    patchs = np.clip(patchs,0,1)
    return patchs    
    
