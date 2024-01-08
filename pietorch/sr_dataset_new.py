import os.path
import torch.utils.data as data
from .srutil import *

class LRGTDataset(data.Dataset):

    def __init__(self,dataroot_GT,dataroot_LR,scale,GT_size,phase='train',use_flip=True,use_rot=True):
        super(LRGTDataset, self).__init__()
        # self.  =
        self.dataroot_GT = dataroot_GT
        self.dataroot_LR = dataroot_LR
        self.scale = scale
        self.GT_size = GT_size
        self.phase = phase
        self.use_flip=use_flip
        self.use_rot=use_rot
        self.paths_LR = None
        self.paths_GT = None
        self.LR_env = None  # environment for lmdb
        self.GT_env = None
        data_type = 'img'
        # read image list from subset list txt
        # read image list from lmdb or image files
        self.GT_env, self.paths_GT = get_image_paths( data_type,  dataroot_GT)
        self.LR_env, self.paths_LR = get_image_paths( data_type,  dataroot_LR)

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LR and self.paths_GT:
            assert len(self.paths_LR) == len(self.paths_GT), \
                'GT and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_GT))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        GT_path, LR_path = None, None
        scale = self.scale
        GT_size = self.GT_size

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT =  read_img(self.GT_env, GT_path)

        # modcrop in the validation / test phase
        if self.phase != 'train':
            img_GT =  modcrop(img_GT, scale)

        # change color space if necessary
        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR =  read_img(self.LR_env, LR_path)
        else:  
            # randomly scale during training
            if self.phase== 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LR =  imresize_np(img_GT, 1 / scale, True)
            if img_LR.ndim == 2: img_LR = np.expand_dims(img_LR, axis=2)

        if self.phase == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2: img_GT = np.expand_dims(img_GT, axis=2)
                # using matlab imresize
                img_LR =  imresize_np(img_GT, 1 / scale, True)
                if img_LR.ndim == 2: img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            LR_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LR, img_GT =  augment([img_LR, img_GT], self.use_flip, self.use_rot)

        # change color space if necessary

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = GT_path
        return img_LR, img_GT, GT_path

    def __len__(self):
        return len(self.paths_GT)
