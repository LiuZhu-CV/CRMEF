import functools

import torch
import torch.nn as nn
# from image_MEF_Fusion_deform.operations_simple import *
from operations_simple import *

import torch.nn.functional as F

class MixedOp(nn.Module):

  def __init__(self, C, primitive):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    kernel = 3
    dilation = 1
    if primitive.find('attention') != -1:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
    else:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
        dilation = int(primitive.split('_')[2])
    print(name, kernel, dilation)
    self._op = OPS[name](C, kernel, dilation, False)

  def forward(self, x):
    return self._op(x)

# 两种网络结构Chain

class Cell_Fusion(nn.Module):

  def __init__(self, C,type,concat):
    super(Cell_Fusion, self).__init__()
    op_names, indices = zip(*type)
    concat = concat
    self._compile(C, op_names, indices, concat)
    self.stem_1 = nn.Sequential(
        nn.Conv2d(64, 64, 3, padding=1, bias=False)
    )
    self.stem_2 = nn.Sequential(
        nn.Conv2d(128, 64, 3, padding=1, bias=False)
    )
    self.sigmoid = nn.Sigmoid()
  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      print(name,index)
      op = MixedOp(C,name)
      self._ops += [op]
    self._indices = indices

  def forward(self, s1,s2,s3):
    s1_0 = self.stem_1(s1)
    s2_0 = self.stem_1(s2)
    s3_0 = self.stem_1(s3)
    s_sum = s1+s2+s3

    offset = 0
    s1 = s_sum
    for i in range(self._steps):
      s1 = self._ops[offset](s1)
      offset += 1
    s_at = s1

    s1_1 = torch.cat([s_at,s1_0],dim=1)
    s2_1 = torch.cat([s_at, s2_0], dim=1)
    s3_1 = torch.cat([s_at, s3_0], dim=1)

    s1_2_1 = self.sigmoid(self.stem_2(s1_1))* s1_0
    s1_2_2 = self.sigmoid(self.stem_2(s2_1))* s2_0
    s1_2_3 = self.sigmoid(self.stem_2(s3_1))* s3_0
    return s1_2_1,s1_2_2,s1_2_3

class Cell_Fusion2(nn.Module):

  def __init__(self, C,type,concat):
    super(Cell_Fusion2, self).__init__()
    op_names, indices = zip(*type)
    concat = concat
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      print(name,index)
      op = MixedOp(C,name)
      self._ops += [op]
    self._indices = indices

  def forward(self, inp_features):
    # skip connection
    offset = 0
    offeset_correction = 0
    s1 = inp_features
    state = []
    for i in range(self._steps):
      s1 = self._ops[offset](s1)
      offset += 1
      for index in range(i):
          # print(index,i,'test'*20)
          s1 += state[index]
          offeset_correction+=1
      state.append(s1)
    return inp_features+s1



def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class PCD_Align_relaxed(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, genotype, nf=64, groups=8):
      super(PCD_Align_relaxed, self).__init__()
      # L3: level 3, 1/4 spatial size
      op_names, indices = zip(*genotype.c)
      self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
      self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
      self.L3_dcnpack = OPS[op_names[0]](nf,nf)
      # extra_offset_mask=True)
      # L2: level 2, 1/2 spatial size
      self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
      self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
      self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
      self.L2_dcnpack = OPS[op_names[1]](nf,nf)
      # extra_offset_mask=True)
      self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
      # L1: level 1, original spatial size
      self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
      self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
      self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
      self.L1_dcnpack = OPS[op_names[2]](nf,nf)
      # extra_offset_mask=True)
      self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
      # Cascading DCN
      self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
      self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

      self.cas_dcnpack = OPS[op_names[3]](nf,nf)
      # extra_offset_mask=True)

      self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
      # print('x1', np.shape(nbr_fea_l), 'x2', np.shape(nbr_fea_l))

      '''align other neighboring frames to the reference frame in the feature level
      nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
      '''
      # L3
      L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
      L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
      L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
      L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
      # L2
      L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
      L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
      L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
      L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
      L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
      L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
      L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
      L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
      # L1
      L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
      L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
      L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
      L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
      L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
      L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
      L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
      L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
      # Cascading
      offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
      offset = self.lrelu(self.cas_offset_conv1(offset))
      offset = self.lrelu(self.cas_offset_conv2(offset))
      L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

      return L1_fea


class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
class Pyramid(nn.Module):
  def __init__(self, in_channels=3, n_feats=64):
    super(Pyramid, self).__init__()
    self.in_channels = in_channels
    self.n_feats = n_feats
    num_feat_extra = 1

    self.conv1 = nn.Sequential(
      nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
    layers = []
    for _ in range(num_feat_extra):
      layers.append(ResidualBlockNoBN(n_feats))
    self.feature_extraction = nn.Sequential(*layers)
    self.downsample1 = nn.Sequential(
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
    self.downsample2 = nn.Sequential(
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
  def forward(self, x):
    x_in = self.conv1(x)
    x1 = self.feature_extraction(x_in)
    x2 = self.downsample1(x1)
    x3 = self.downsample2(x2)
    return [x1, x2, x3]

class Pyramid_light(nn.Module):
  def __init__(self, in_channels=3, n_feats=64):
    super(Pyramid_light, self).__init__()
    self.in_channels = in_channels
    self.n_feats = n_feats
    num_feat_extra = 1

    self.conv1 = nn.Sequential(
      nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
    layers = []
    for _ in range(num_feat_extra):
      layers.append(ResidualBlockNoBN(n_feats))
    self.feature_extraction = nn.Sequential(*layers)
    # self.downsample1 = nn.Sequential(
    #   nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
    #   nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #   nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
    #   nn.LeakyReLU(negative_slope=0.1, inplace=True)
    # )
    # self.downsample2 = nn.Sequential(
    #   nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
    #   nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #   nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
    #   nn.LeakyReLU(negative_slope=0.1, inplace=True)
    # )
  def forward(self, x):
    x_in = self.conv1(x)
    x1 = self.feature_extraction(x_in)
    # x2 = self.downsample1(x1)
    # x3 = self.downsample2(x2)
    return x1
class Pyramid2(nn.Module):
  def __init__(self, in_channels=3, n_feats=64):
    super(Pyramid2, self).__init__()
    self.in_channels = in_channels
    self.n_feats = n_feats
    num_feat_extra = 1

    # self.conv1 = nn.Sequential(
    #   nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
    #   nn.LeakyReLU(negative_slope=0.1, inplace=True)
    # )
    layers = []
    for _ in range(num_feat_extra):
      layers.append(ResidualBlockNoBN(n_feats))
    self.feature_extraction = nn.Sequential(*layers)
    self.downsample1 = nn.Sequential(
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
    self.downsample2 = nn.Sequential(
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True),
      nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
  def forward(self, x):
    x1 = self.feature_extraction(x)
    x2 = self.downsample1(x1)
    x3 = self.downsample2(x2)
    return [x1, x2, x3]
  
  
class AttenionNet(torch.nn.Module):
    def __init__(self):
        super(AttenionNet, self).__init__()

        self.fe1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.fe2 = torch.nn.Conv2d(64, 64, 3, 1, 1)

        self.sAtt_1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = torch.nn.Conv2d(64 * 2, 64, 1, 1, bias=True)
        self.sAtt_3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.sAtt_4 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_5 = torch.nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.sAtt_L1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_L2 = torch.nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.sAtt_L3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, alignedframe):

        # feature extraction
        att = self.lrelu(self.fe1(alignedframe))
        att = self.lrelu(self.fe2(att))

        # spatial attention
        att = self.lrelu(self.sAtt_1(att))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L,
                              size=[att.size(2), att.size(3)],
                              mode='bilinear', align_corners=False)


        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att,
                            size=[alignedframe.size(2), alignedframe.size(3)],
                            mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        # att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        return att


class AttenionNet_relaxed(torch.nn.Module):
    def __init__(self, genotype):
        super(AttenionNet_relaxed, self).__init__()
        op_names, indices = zip(*genotype.m)
        # self.dc = self.distilled_channels = self.channel  # // 2
        # self.rc = self.remaining_channels = self.channel
        # self.c1_d = OPS[op_names[0]](self.channel, self.dc)
        # self.c1_r = OPS[op_names[1]](self.channel, self.rc)
        self.fe1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.fe2 = OPS[op_names[0]](64, 64)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = OPS[op_names[1]](64*2, 64)
        self.sAtt_4 = OPS[op_names[2]](64, 64)
        self.sAtt_5 = torch.nn.Conv2d(64, 3, 3, 1, 1)
        self.sAtt_L1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.lrelu =torch.nn.PReLU()
    def forward(self, alignedframe):
        # feature extraction
        att = self.lrelu(self.fe1(alignedframe))
        att = self.lrelu(self.fe2(att))
        # spatial attention
        att = self.lrelu(att)
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att = self.lrelu(self.sAtt_4(att)) + att
        att_ = F.interpolate(att,
                            size=[alignedframe.size(2), alignedframe.size(3)],
                            mode='bilinear', align_corners=False)
        att = self.sAtt_5(att_)
        att = torch.sigmoid(att)
        return att*alignedframe


class illuminationNet(nn.Module):
    def __init__(self,genotype):
        super(illuminationNet, self).__init__()
        op_names, indices = zip(*genotype.f)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.PReLU()

        self.conv4 = OPS[op_names[0]](128, 128)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.PReLU()

        self.conv5 = OPS[op_names[1]](128, 128)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.PReLU()

        self.conv8 = OPS[op_names[2]](128, 128)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.PReLU()

        self.conv25 = OPS[op_names[3]](128, 128)
        self.bn25 = nn.BatchNorm2d(128)
        self.relu9 = nn.PReLU()
        # 26
        self.conv26 = nn.Conv2d(128, 3, 1, stride=1, padding=0)
        self.bn26 = nn.BatchNorm2d(3)

        
        self.tanh = nn.Sigmoid()

    def forward(self, input):
        input = torch.tensor(input)
        input_ = self.conv1(input)
        x = self.bn3(self.conv3(input_))
        x = self.relu3(x)
       
        x = self.bn4(self.conv4(x))
        x = self.relu4(x)
        res4 = x
        x = self.bn5(self.conv5(x))
        x = self.relu5(x + res4)
        x = self.bn8(self.conv8(x))
        x = self.relu8(x)
        res7 = x
        x = self.bn25(self.conv25(x))
        x = self.relu9(x + res7)
        latent = self.conv26(x)

     
        latent = self.tanh(latent)
        output = input / (latent + 0.00001)
        return output, latent


class illuminationNet_light(nn.Module):
    def __init__(self, C, genotype):
        super(illuminationNet_light, self).__init__()
        op_names, indices = zip(*genotype.f)
        self.conv1 = nn.Conv2d(3, C//2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(C//2, C, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(C)
        self.relu3 = nn.PReLU()

        self.conv4 = OPS[op_names[0]](C, C)
        self.bn4 = nn.BatchNorm2d(C)
        self.relu4 = nn.PReLU()

        self.conv5 = OPS[op_names[1]](C, C)
        self.bn5 = nn.BatchNorm2d(C)
        self.relu5 = nn.PReLU()

        self.conv8 = OPS[op_names[2]](C, C)
        self.bn8 = nn.BatchNorm2d(C)
        self.relu8 = nn.PReLU()

        self.conv25 = OPS[op_names[3]](C, C)
        self.bn25 = nn.BatchNorm2d(C)
        self.relu9 = nn.PReLU()
        # 26
        self.conv26 = nn.Conv2d(C, 3, 1, stride=1, padding=0)
        self.bn26 = nn.BatchNorm2d(3)

        self.tanh = nn.Sigmoid()

    def forward(self, input):
        input = torch.tensor(input)
        input_ = self.conv1(input)
        x = self.bn3(self.conv3(input_))
        x = self.relu3(x)

        x = self.bn4(self.conv4(x))
        x = self.relu4(x)
        res4 = x
        x = self.bn5(self.conv5(x))
        x = self.relu5(x + res4)
        x = self.bn8(self.conv8(x))
        x = self.relu8(x)
        res7 = x
        x = self.bn25(self.conv25(x))
        x = self.relu9(x + res7)
        latent = self.conv26(x)

        latent = self.tanh(latent)
        output = input / (latent + 0.00001)
        return output, latent

class illuminationNet_deform(nn.Module):
    def __init__(self, genotype):
        super(illuminationNet_deform, self).__init__()
        op_names, indices = zip(*genotype.f)
        self.pyramid = Pyramid(n_feats=64)
        self.deform = PCD_Align_relaxed(genotype=genotype,)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.PReLU()

        self.conv4 = OPS[op_names[0]](64, 64)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.PReLU()

        self.conv5 = OPS[op_names[1]](64, 64)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.PReLU()

        self.conv8 = OPS[op_names[2]](64, 64)
        self.bn8 = nn.BatchNorm2d(64)
        self.relu8 = nn.PReLU()

        self.conv25 = OPS[op_names[3]](64, 64)
        self.bn25 = nn.BatchNorm2d(64)
        self.relu9 = nn.PReLU()
        # 26
        self.conv26 = nn.Conv2d(64, 3, 1, stride=1, padding=0)
        self.bn26 = nn.BatchNorm2d(3)

        self.tanh = nn.Sigmoid()

    def forward(self, x1,x2):
        F1 = self.pyramid(x1)
        F2 = self.pyramid(x2)
        F1_ = self.deform(F1, F2)
        F2_ = F2[0]
        input = F1_ + F2_
        x = self.bn3(self.conv3(input))
        input_ = self.relu3(x)

        x = self.bn4(self.conv4(input_))
        x = self.relu4(x)
        res4 = x
        x = self.bn5(self.conv5(x))
        x = self.relu5(x + res4)
        x = self.bn8(self.conv8(x))
        x = self.relu8(x)
        res7 = x
        x = self.bn25(self.conv25(x))
        x = self.relu9(x + res7)
        latent = self.conv26(input*x)

        latent = self.tanh(latent)
        output = latent
        return output

class illuminationNet_deform3(nn.Module):
    def __init__(self, genotype):
        super(illuminationNet_deform3, self).__init__()
        op_names, indices = zip(*genotype.f)
        self.pyramid = Pyramid(n_feats=64)
        self.deform = PCD_Align_relaxed(genotype=genotype,)
        self.conv3 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(3)
        self.relu3 = nn.PReLU()
        #
        # self.conv4 = OPS[op_names[0]](64, 64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.PReLU()
        #
        # self.conv5 = OPS[op_names[1]](64, 64)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.relu5 = nn.PReLU()
        #
        # self.conv8 = OPS[op_names[2]](64, 64)
        # self.bn8 = nn.BatchNorm2d(64)
        # self.relu8 = nn.PReLU()
        #
        # self.conv25 = OPS[op_names[3]](64, 64)
        # self.bn25 = nn.BatchNorm2d(64)
        # self.relu9 = nn.PReLU()
        # # 26
        # self.conv26 = nn.Conv2d(64, 3, 1, stride=1, padding=0)
        # self.bn26 = nn.BatchNorm2d(3)

        self.tanh = nn.Sigmoid()

    def forward(self, x1,x2):
        F1 = self.pyramid(x1)
        F2 = self.pyramid(x2)
        F1_ = self.deform(F1, F2)
        # F2_ = F2[0]
        # input = F1_ + F2_
        x = self.bn3(self.conv3(F1_))
        input_ = self.relu3(x)


        latent = self.tanh(input_)
        output = latent
        return output,x2

class illuminationNet_deform2(nn.Module):
    def __init__(self, genotype):
        super(illuminationNet_deform2, self).__init__()
        op_names, indices = zip(*genotype.f)
        # self.pyramid = Pyramid(n_feats=64)
        # self.deform = PCD_Align_relaxed(genotype=genotype,)
        self.conv3 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.PReLU()

        self.conv4 = OPS[op_names[0]](64, 64)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.PReLU()

        self.conv5 = OPS[op_names[1]](64, 64)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.PReLU()

        self.conv8 = OPS[op_names[2]](64, 64)
        self.bn8 = nn.BatchNorm2d(64)
        self.relu8 = nn.PReLU()

        self.conv25 = OPS[op_names[3]](64, 64)
        self.bn25 = nn.BatchNorm2d(64)
        self.relu9 = nn.PReLU()
        # 26
        self.conv26 = nn.Conv2d(64, 3, 1, stride=1, padding=0)
        self.bn26 = nn.BatchNorm2d(3)

        self.tanh = nn.Sigmoid()

    def forward(self, x1,x2):
        # F1 = self.pyramid(x1)
        # F2 = self.pyramid(x2)
        # F1_ = self.deform(F1, F2)
        # F2_ = F2[0]
        input = x1 + x2
        x = self.bn3(self.conv3(input))
        input_ = self.relu3(x)

        x = self.bn4(self.conv4(input_))
        x = self.relu4(x)
        res4 = x
        x = self.bn5(self.conv5(x))
        x = self.relu5(x + res4)
        x = self.bn8(self.conv8(x))
        x = self.relu8(x)
        res7 = x
        x = self.bn25(self.conv25(x))
        x = self.relu9(x + res7)
        latent = self.conv26(x)

        latent = self.tanh(latent)
        output = input / (latent + 0.00001)
        return output

class AHDR_MEF_deform(nn.Module):
    def __init__(self, nChannel,genotypes,layer =2):
        super(AHDR_MEF_deform, self).__init__()
        self.nChannel = nChannel
        self.layers = layer
        self.attention_under = nn.ModuleList()
        self.attention_over = nn.ModuleList()
        for index in range(layer):
            self.attention_under.append(AttenionNet_relaxed(genotype=genotypes))
            self.attention_over.append(AttenionNet_relaxed(genotype=genotypes))
        self.aggregation = illuminationNet_deform(genotype=genotypes)
    def forward(self, x1, x2):
        for index  in range(self.layers):
           x1 = self.attention_under[index](x1)
           x2 = self.attention_over[index](x2)
        # x_robust = x1 + x2
        x_final = self.aggregation(x1,x2)
        return x1, x_final

class AHDR_MEF_deform3(nn.Module):
    def __init__(self, nChannel,genotypes,layer =2):
        super(AHDR_MEF_deform3, self).__init__()
        self.nChannel = nChannel
        self.layers = layer
        self.attention_under = nn.ModuleList()
        self.attention_over = nn.ModuleList()
        for index in range(layer):
            self.attention_under.append(AttenionNet_relaxed(genotype=genotypes))
            self.attention_over.append(AttenionNet_relaxed(genotype=genotypes))
        self.aggregation = illuminationNet_deform2(genotype=genotypes)
        self.aggregation3 = illuminationNet_deform3(genotype=genotypes)
    def forward(self, x1, x2):
        x1,x2 = self.aggregation3(x1,x2)
        for index  in range(self.layers):
           x1 = self.attention_under[index](x1)
           x2 = self.attention_over[index](x2)
        # x_robust = x1 + x2
        x_final = self.aggregation(x1,x2)
        return x1, x_final


class AHDR_MEF_deform2(nn.Module):
    def __init__(self, nChannel,genotypes,layer =2):
        super(AHDR_MEF_deform2, self).__init__()
        self.nChannel = nChannel
        self.layers = layer
        self.attention_under = nn.ModuleList()
        self.attention_over = nn.ModuleList()
        for index in range(layer):
            self.attention_under.append(AttenionNet_relaxed(genotype=genotypes))
            self.attention_over.append(AttenionNet_relaxed(genotype=genotypes))
        self.aggregation = illuminationNet_deform2(genotype=genotypes)
    def forward(self, x1, x2):
        for index  in range(self.layers):
           x1 = self.attention_under[index](x1)
           x2 = self.attention_over[index](x2)
        # x_robust = x1 + x2
        x_final = self.aggregation(x1,x2)
        return x1, x_final

class AHDR_MEF(nn.Module):
    def __init__(self, nChannel,genotypes,layer =2):
        super(AHDR_MEF, self).__init__()
        self.nChannel = nChannel
        self.layers = layer
        self.attention_under = nn.ModuleList()
        self.attention_over = nn.ModuleList()
        for index in range(layer):
            self.attention_under.append(AttenionNet_relaxed(genotype=genotypes))
            self.attention_over.append(AttenionNet_relaxed(genotype=genotypes))
        self.aggregation = illuminationNet(genotype=genotypes)
    def forward(self, x1, x2):
        arr = []
        for index  in range(self.layers):
           x1 = self.attention_under[index](x1)
           x2 = self.attention_over[index](x2)
           # att1 = (att1-att1.min())/(att1.max()-att1.min())
           # att2 =( att2 -att2.min())/ (att2.max() - att2.min())
           arr.append(x1)
           arr.append(x2)
        x_robust = x1 + x2
        x_final, att = self.aggregation(x_robust)
        arr.append(att)
        return x_robust,x_final


class AHDR_MEF_light(nn.Module):
    def __init__(self, nChannel,genotypes,layer =2,C=64):
        super(AHDR_MEF_light, self).__init__()
        self.nChannel = nChannel
        self.layers = layer
        self.attention_under = nn.ModuleList()
        self.attention_over = nn.ModuleList()
        for index in range(layer):
            self.attention_under.append(AttenionNet_relaxed(genotype=genotypes))
            self.attention_over.append(AttenionNet_relaxed(genotype=genotypes))
        self.aggregation = illuminationNet_light(C=C,genotype=genotypes)
    def forward(self, x1, x2):
        arr = []
        for index  in range(self.layers):
           x1 = self.attention_under[index](x1)
           x2 = self.attention_over[index](x2)
           # att1 = (att1-att1.min())/(att1.max()-att1.min())
           # att2 =( att2 -att2.min())/ (att2.max() - att2.min())
           arr.append(x1)
           arr.append(x2)
        x_robust = x1 + x2
        x_final, att = self.aggregation(x_robust)
        arr.append(att)
        return x_robust,x_final



class AttentionModule_1(nn.Module):

    def __init__(self,  genotype,channel=16):
        super(AttentionModule_1, self).__init__()

        self.stride = 1
        self.channel = channel

        op_names, indices = zip(*genotype.m)
        self.dc = self.distilled_channels = self.channel  # // 2
        self.rc = self.remaining_channels = self.channel
        self.c1_d = OPS[op_names[0]](self.channel, self.dc)
        self.c1_r = OPS[op_names[1]](self.channel, self.rc)
        self.relu = nn.LeakyReLU()
        self.conv = conv_layer(channel*2,channel,3)
    def forward(self, F1_, F2_):
        F1_i = torch.cat((F1_, F2_), 1)
        F1_i = (self.conv(F1_i))
        F1_i = self.relu(self.c1_d(F1_i))
        F1_i = (self.c1_r(F1_i))
        F1_A = nn.functional.sigmoid(F1_i)
        return F1_ * F1_A

class AttentionModule_2(nn.Module):

    def __init__(self,  genotype,channel=16):
        super(AttentionModule_2, self).__init__()

        self.stride = 1
        self.channel = channel

        op_names, indices = zip(*genotype.m)
        self.dc = self.distilled_channels = self.channel  # // 2
        self.rc = self.remaining_channels = self.channel
        self.c1_d = OPS[op_names[0]](self.channel, self.dc)
        self.c1_r = OPS[op_names[1]](self.channel, self.rc)
        self.relu = nn.LeakyReLU()
        self.conv = conv_layer(channel*2,channel,3)
    def forward(self, F1_, F2_):
        F1_i = torch.cat((F1_, F2_), 1)
        F1_i = (self.conv(F1_i))
        F1_i = self.relu(self.c1_d(F1_i))
        F1_i = (self.c1_r(F1_i))
        F1_A = nn.functional.sigmoid(F1_i)
        return F1_ * F1_A

class make_dilation_dense(nn.Module):
    def __init__(self, opname, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = OPS[opname](nChannels, growthRate)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, genotypes):

        super(DRDB, self).__init__()
        nChannels_ = nChannels
        growthRate = 32
        modules = []
        op_names, indices = zip(*genotypes.f)
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(op_names[i],nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out
import numpy as np
class AHDR(nn.Module):
    def __init__(self, nChannel, nFeat,genotypes):
        super(AHDR, self).__init__()
        self.nChannel = nChannel
        self.nFeat = nFeat

        # F-1
        # self.conv1 = nn.Conv2d(6, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.pcd_r = PCD_Align_relaxed(genotypes, 64)
        self.pyramid_feats = Pyramid(3, 64)

        self.conv2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # DRDBs 3
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # conv
        self.relu = nn.LeakyReLU()
        self.att = AttentionModule_1(genotypes)
        self.att2 = AttentionModule_2(genotypes)
        self.RDB1 = DRDB(nFeat,2, genotypes)
        self.RDB2 = DRDB(nFeat,2, genotypes)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        F1_ = self.pyramid_feats(x1)
        F2_ = self.pyramid_feats(x2)
        F1_ = self.pcd_r(F1_, F2_)
        F2_ = F2_[0]
        F1_ = self.att(F1_,F2_)
        F3_ = self.att2(F2_,F1_)
        F_ = torch.cat((F1_, F3_), 1)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        output = self.conv3(FDF)
        output = self.sigmoid(output)

        return output

class AHDR_2(nn.Module):
    def __init__(self, nChannel, nFeat,genotypes):
        super(AHDR_2, self).__init__()
        self.nChannel = nChannel
        self.nFeat = nFeat

        # F-1
        self.conv1_s = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv1_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        self.conv2_s = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv2_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        # F0
        self.pcd_r = PCD_Align_relaxed(genotypes, 64)
        self.pyramid_feats = Pyramid(3, 64)

        self.conv2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # DRDBs 3
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # conv
        self.relu = nn.LeakyReLU()
        self.att = AttentionModule_1(genotypes)
        self.att2 = AttentionModule_2(genotypes)
        self.RDB1 = DRDB(nFeat,2, genotypes)
        self.RDB2 = DRDB(nFeat,2, genotypes)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        s1 = self.conv1_s(x1)
        s2 = self.conv2_s(x2)
        s1 = self.att(s1, s1)
        s2 = self.att2(s2, s2)
        x1 = self.conv1_f(s1)
        x2 = self.conv2_f(s2)
        F1_ = self.pyramid_feats(x1)
        F2_ = self.pyramid_feats(x2)
        F1_ = self.pcd_r(F1_, F2_)
        F2_ = F2_[0]

        F_ = torch.cat((F1_, F2_), 1)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        output = self.conv3(FDF)
        output = self.sigmoid(output)

        return x1,x2,output


class AHDR_4(nn.Module):
    def __init__(self, nChannel, nFeat,genotypes):
        super(AHDR_4, self).__init__()
        self.nChannel = nChannel
        self.nFeat = nFeat

        # F-1
        self.conv1_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_f = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        # F0
        self.pcd_r = PCD_Align_relaxed(genotypes, 64)
        self.pyramid_feats = Pyramid2(64, 64)

        self.conv2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # DRDBs 3
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # conv
        self.relu = nn.LeakyReLU()
        self.att = AttentionModule_1(genotypes,channel=64)
        self.att2 = AttentionModule_2(genotypes,channel=64)
        self.RDB1 = DRDB(nFeat,2, genotypes)
        self.RDB2 = DRDB(nFeat,2, genotypes)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        s1 = self.conv1_s(x1)
        s2 = self.conv2_s(x2)
        s1 = self.att(s1, s1)
        s2 = self.att2(s2, s2)
        F1_ = self.pyramid_feats(s1)
        F2_ = self.pyramid_feats(s2)

        F1_ = self.pcd_r(F1_, F2_)
        F2_ = F2_[0]

        F_ = torch.cat((F1_, F2_), 1)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        output = self.conv3(FDF)
        output = self.sigmoid(output)

        return x1,x2,output


class AHDR_6(nn.Module):
    def __init__(self, nChannel, nFeat,genotypes):
        super(AHDR_6, self).__init__()
        self.nChannel = nChannel
        self.nFeat = nFeat

        # F-1
        self.conv1_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_f = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        # F0
        self.pcd_r = PCD_Align_relaxed(genotypes, 64)
        self.pyramid_feats = Pyramid2(64, 64)

        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # DRDBs 3
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # conv
        self.relu = nn.LeakyReLU()
        self.att = AttentionModule_1(genotypes,channel=64)
        self.att2 = AttentionModule_2(genotypes,channel=64)
        self.RDB1 = DRDB(nFeat,2, genotypes)
        self.RDB2 = DRDB(nFeat,2, genotypes)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        s1 = self.conv1_s(x1)
        s2 = self.conv2_s(x2)
        s1 = self.att(s1, s1)
        s2 = self.att2(s2, s2)
        F1_ = self.pyramid_feats(s1)
        F2_ = self.pyramid_feats(s2)

        F1_ = self.pcd_r(F1_, F2_)
        F2_ = F2_[0]

        F_ = F1_ + F2_
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        output = self.sigmoid(FGF)

        return x1,x2,output
    
class AHDR_4_light(nn.Module):
    def __init__(self, nChannel, nFeat,genotypes):
        super(AHDR_4_light, self).__init__()
        self.nChannel = nChannel
        self.nFeat = nFeat

        # F-1
        self.conv1_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_f = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # DRDBs 3
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # conv
        self.relu = nn.LeakyReLU()
        self.att = AttentionModule_1(genotypes,channel=64)
        self.att2 = AttentionModule_2(genotypes,channel=64)
        self.RDB1 = DRDB(nFeat,2, genotypes)
        self.RDB2 = DRDB(nFeat,2, genotypes)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        s1 = self.conv1_s(x1)
        s2 = self.conv2_s(x2)
        s1 = self.att(s1, s1)
        s2 = self.att2(s2, s2)
        F_ = torch.cat((s1, s2), 1)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        output = self.sigmoid(FGF)
        return x1,x2,output
    
class AHDR_5(nn.Module):
    def __init__(self, nChannel, nFeat,genotypes):
        super(AHDR_5, self).__init__()
        self.nChannel = nChannel
        self.nFeat = nFeat

        # F-1
        self.conv1_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_f = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_s = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.conv2_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        # F0
        self.pcd_r = PCD_Align_relaxed(genotypes, 64)
        self.pyramid_feats = Pyramid2(64, 64)

        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # DRDBs 3
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # conv
        self.relu = nn.LeakyReLU()
        self.att = AttentionModule_1(genotypes,channel=64)
        self.att2 = AttentionModule_2(genotypes,channel=64)
        self.RDB1 = DRDB(nFeat,2, genotypes)
        self.RDB2 = DRDB(nFeat,2, genotypes)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        s1 = self.conv1_s(x1)
        s2 = self.conv2_s(x2)
        s1 = self.att(s1, s1)
        s2 = self.att2(s2, s2)
        F1_ = self.pyramid_feats(s1)
        F2_ = self.pyramid_feats(s2)
        F1_ = self.pcd_r(F1_, F2_)
        F2_ = F2_[0]
        F_ = F1_ + F2_
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        output = self.conv3(FDF)
        output = self.sigmoid(output)

        return x1,x2,output
class AHDR_3(nn.Module):
    def __init__(self, nChannel, nFeat,genotypes):
        super(AHDR_3, self).__init__()
        self.nChannel = nChannel
        self.nFeat = nFeat

        # F-1
        self.conv1_s = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv1_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        self.conv2_s = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv2_f = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        # F0
        # self.pcd_r = PCD_Align_relaxed(genotypes, 64)
        self.pyramid_feats = Pyramid_light(3, nFeat)

        self.conv2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        # DRDBs 3
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # conv
        self.relu = nn.LeakyReLU()
        self.att = AttentionModule_1(genotypes)
        self.att2 = AttentionModule_2(genotypes)
        self.RDB1 = DRDB(nFeat,2, genotypes)
        self.RDB2 = DRDB(nFeat,2, genotypes)
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        s1 = self.conv1_s(x1)
        s2 = self.conv2_s(x2)
        s1 = self.att(s1, s1)
        s2 = self.att2(s2, s2)
        x1 = self.conv1_f(s1)
        x2 = self.conv2_f(s2)
        F1_ = self.pyramid_feats(x1)
        F2_ = self.pyramid_feats(x2)
        # F1_ = self.pcd_r(F1_, F2_)
        # F2_ = F2_[0]

        F_ = torch.cat((F1_, F2_), 1)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        FF = torch.cat((F_1, F_2), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_
        output = self.conv3(FDF)
        output = self.sigmoid(output)

        return x1,x2,output

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


