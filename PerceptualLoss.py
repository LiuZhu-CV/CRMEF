import torch
import torch.nn.functional as F
from pietorch.pytorch_ssim import ssim as ssim  #ssim loss AG and other features
import torch.nn as nn

# --- Perceptual loss network  --- #
# SSIM MSE
# TVLOSS
def get_tv(tensor):
  N, C, H, W = tensor.size()
  f = tensor[:, :, :-1, :-1]
  g = tensor[:, :, :-1, 1:]
  h = tensor[:, :, 1:, :-1]
  tv_= (f - g) ** 2. + (f - h) ** 2.
  return tv_

# Feature get weights

class LaplacianOperator(torch.nn.Module):
    def __init__(self):
        super(LaplacianOperator, self).__init__()
        K = torch.tensor([[1 / 8, 1 / 8, 1 / 8],
                          [1 / 8, -1, 1 / 8],
                          [1 / 8, 1 / 8, 1 / 8]], requires_grad=False,
                         dtype=torch.float).view((1, 1, 3, 3))
        self.register_buffer('K', K)

    def forward(self, X):
        b,c,h,w = X.shape

        for i in range(c):
            X_f = F.conv2d(torch.unsqueeze(X[:,i,:,:],dim=1), self.K)
            if i==0:
                fgs = X_f
            else:
                fgs = torch.cat([fgs,X_f],dim=1)


        return fgs


class VGGBasedSSIMMSEloss(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGGBasedSSIMMSEloss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = { # only use the shallow features
            '1': "relu1_2",
            '3': "relu2_2",
            '6': "relu3_1",
            '8': "relu3_1"
        }
        self.lap = LaplacianOperator()
    def output_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # output[self.layer_name_mapping[name]] = x
                feature_norm = torch.norm(self.lap(x))
                output.append(feature_norm)
            # get two values
            # first get Lap and normalize --> softmax
            # Then obtain the weights
        return sum(output) / len(output)


    def forward(self, lr, vis,output):
        lr = torch.cat([lr,lr,lr],dim=1)
        vis = torch.cat([vis, vis, vis], dim=1)
        output = torch.cat([output, output, output], dim=1)
        w1 = self.output_features(lr)/3500
        w2 = self.output_features(vis)/3500
        loss= torch.tensor([w1,w2]).cuda()
        weight = F.softmax(loss,dim=0).cuda()
        # for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
        #     loss.append(F.mse_loss(dehaze_feature, gt_feature))
        # get SSIM Loss
        SSIM_loss = weight[0] *(1-ssim(lr,output)) + weight[1] * (1-ssim(vis,output))
        # get MSE loss all
        MSE_loss = weight[0] *F.mse_loss(lr,output) + weight[1]*F.mse_loss(vis,output)
        return SSIM_loss + MSE_loss*20

class VGGBasedSSIMMSEloss2(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGGBasedSSIMMSEloss2, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = { # only use the shallow features
            '1': "relu1_2",
            '3': "relu2_2",
            '6': "relu3_1",
            '8': "relu3_1"
        }
        self.lap = LaplacianOperator()
    def output_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # output[self.layer_name_mapping[name]] = x
                feature_norm = torch.norm(self.lap(x))
                output.append(feature_norm)
            # get two values
            # first get Lap and normalize --> softmax
            # Then obtain the weights
        return sum(output) / len(output)


    def forward(self, lr, vis,output):
        lr = torch.cat([lr,lr,lr],dim=1)
        vis = torch.cat([vis, vis, vis], dim=1)
        output = torch.cat([output, output, output], dim=1)
        # w1 = self.output_features(lr)/3500
        # w2 = self.output_features(vis)/3500
        # loss= torch.tensor([w1,w2]).cuda()
        # weight = F.softmax(loss,dim=0).cuda()
        # for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
        #     loss.append(F.mse_loss(dehaze_feature, gt_feature))
        # get SSIM Loss
        SSIM_loss = 0.38 *(1-ssim(lr,output)) + 0.62* (1-ssim(vis,output))
        # get MSE loss all
        MSE_loss = 0.6 *F.l1_loss(lr,output) + 0.4*F.l1_loss(vis,output)
        return SSIM_loss + MSE_loss*10
class VGGBasedSSIMloss(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGGBasedSSIMloss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = { # only use the shallow features
            '1': "relu1_2",
            '3': "relu2_2",
            '6': "relu3_1",
            '8': "relu3_1"
        }
        self.lap = LaplacianOperator()
    def output_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # output[self.layer_name_mapping[name]] = x
                feature_norm = torch.norm(self.lap(x))
                output.append(feature_norm)
            # get two values
            # first get Lap and normalize --> softmax
            # Then obtain the weights
        return sum(output) / len(output)


    def forward(self, lr, vis,output):
        lr = torch.cat([lr,lr,lr],dim=1)
        vis = torch.cat([vis, vis, vis], dim=1)
        output = torch.cat([output, output, output], dim=1)
        w1 = self.output_features(lr)/3500
        w2 = self.output_features(vis)/3500
        loss= torch.tensor([w1,w2]).cuda()
        weight = F.softmax(loss,dim=0).cuda()
        # for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
        #     loss.append(F.mse_loss(dehaze_feature, gt_feature))
        # get SSIM Loss
        SSIM_loss =  (1-ssim(vis,output))
        # get MSE loss all
        MSE_loss = F.mse_loss(lr,output)
        return SSIM_loss + MSE_loss*15

class VGGBasedSSIMMSEloss3(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGGBasedSSIMMSEloss3, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = { # only use the shallow features
            '1': "relu1_2",
            '3': "relu2_2",
            '6': "relu3_1",
            '8': "relu3_1"
        }
        self.lap = LaplacianOperator()
    def output_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # output[self.layer_name_mapping[name]] = x
                feature_norm = torch.norm(self.lap(x))
                output.append(feature_norm)
            # get two values
            # first get Lap and normalize --> softmax
            # Then obtain the weights
        return sum(output) / len(output)


    def forward(self, lr,output):
        lr = torch.cat([lr,lr,lr],dim=1)
        output = torch.cat([output, output, output], dim=1)

        SSIM_loss = (1-ssim(lr,output))
        # get MSE loss all
        MSE_loss = F.mse_loss(lr,output)
        return SSIM_loss + MSE_loss*20

class VGGBasedSSIMMSEloss_1(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGGBasedSSIMMSEloss_1, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = { # only use the shallow features
            '8': "relu1_2",
            '11': "relu2_2",
            '15': "relu3_1"
        }
        self.lap = LaplacianOperator()
    def output_features(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                # output[self.layer_name_mapping[name]] = x
                feature_norm = torch.norm(self.lap(x))
                output.append(feature_norm)
            # get two values

            # first get Lap and normalize --> softmax
            # Then obtain the weights
        return sum(output) / len(output)


    def forward(self, lr, vis,output):
        lr = torch.cat([lr,lr,lr],dim=1)
        vis = torch.cat([vis, vis, vis], dim=1)

        output = torch.cat([output, output, output], dim=1)
        # lr = torch.nn.functional.interpolate(lr, size=(224, 224))
        # vis = torch.nn.functional.interpolate(vis, size=(224, 224))
        # image = torch.nn.functional.interpolate(image, size=(224, 224))
        w1 = self.output_features(lr)/1500
        w2 = self.output_features(vis)/1500
        loss= torch.tensor([w1,w2]).cuda()
        weight = F.softmax(loss,dim=0).cuda()

        # for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
        #     loss.append(F.mse_loss(dehaze_feature, gt_feature))
        # get SSIM Loss
        SSIM_loss = weight[0] *(1-ssim(lr,output)) + weight[1] * (1-ssim(vis,output))
        # get MSE loss all

        MSE_loss = weight[0] *F.mse_loss(lr,output) + weight[1]*F.mse_loss(vis,output)
        return SSIM_loss + MSE_loss*20
import torchvision

class LossNetwork_per(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork_per, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        # dehaze = torch.cat([dehaze,dehaze,dehaze],dim=1)
        # gt = torch.cat([gt,gt,gt],dim=1)
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))
        return sum(loss)/len(loss)



class Sobel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        G = torch.cat([G,G,G],dim=1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class new_loss_sobel(torch.nn.Module):
    def __init__(self):
        super(new_loss_sobel, self).__init__()
        self.L1loss = torch.nn.L1Loss().cuda()
        self.sobel =  LaplacianOperator()
    def forward(self, img, gt):
        mse_ = self.L1loss(img,gt)
        sobel_ = self.L1loss(self.sobel(img), self.sobel(gt))
        return mse_ + 0.5*sobel_

