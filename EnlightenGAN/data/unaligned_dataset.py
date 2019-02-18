import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from EnlightenGAN.data.base_dataset import BaseDataset, get_transform
from EnlightenGAN.data.image_folder import make_dataset
import random
from PIL import Image
import PIL
from pdb import set_trace as st
import numpy as np
from skimage import color


def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class UnalignedDataset(BaseDataset):
    def _reinit_A_paths(self):
        self.A_paths = self.pos_names# + np.random.choice(self.neg_names_all, int(948/(10/1)), replace=False).tolist()
        random.shuffle(self.A_paths)
        self.B_paths = list(self.A_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        ##############################
        self.dir_A = os.path.join(opt.dataroot)#, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot)#, opt.phase + 'B')
        # if not 'images' in self.opt.name:
        #     self.dir_A = os.path.join("/ssd1/chenwy/bdd100k/seg_luminance/0_75/", opt.phase)
        #     self.dir_B = os.path.join("/ssd1/chenwy/bdd100k/seg_luminance/100_105/", opt.phase)
        # else:
        #     self.dir_A = os.path.join("/ssd1/chenwy/bdd100k/images_luminance/100k/0_75/", opt.phase)
        #     self.dir_B = os.path.join("/ssd1/chenwy/bdd100k/images_luminance/100k/100_105/", opt.phase)
        ##############################

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)

        ##### load image2reward to resample dataset ############################
        # image2reward = np.load("/home/chenwy/DynamicLightEnlighten/image2reward.npy").item()
        # self.pos = []; self.pos_names = []; self.neg_names_all = []
        # for k, v in image2reward.items():
        #     if v > 0:
        #         self.pos.append(v)
        #         self.pos_names.append(k)
        #     elif v < 0:
        #         self.neg_names_all.append(k)
        # self.pos_names = [k for v,k in sorted(zip(self.pos, self.pos_names), reverse=True)]
        # self._reinit_A_paths()
        #################################

        self.low_range = range(0, 75)
        self.high_range = range(100, 110)
        self.N_TRY = 20

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B % self.B_size]

        A_image = Image.open(A_path).convert('RGB')
        B_image = Image.open(B_path).convert('RGB')
        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray

        w, h = A_image.size
        n_try = 0
            x1 = random.randint(0, w - self.opt.fineSize)
            y1 = random.randint(0, h - self.opt.fineSize)
            A_img = A_image.crop((x1, y1, x1+self.opt.fineSize, y1+self.opt.fineSize))
            B_img = B_image.crop((x1, y1, x1+self.opt.fineSize, y1+self.opt.fineSize))
            A_npy = np.array(A_img)
            B_npy = np.array(B_img)

            r,g,b = A_npy[:, :, 0], A_npy[:, :, 1], A_npy[:, :, 2]
            value_A = (0.299*r+0.587*g+0.114*b) / 255.
            value_A = np.sort(value_A.flatten())
            length = value_A.shape[0]
            value_A = value_A[int(np.round(length * 0.1)) : int(np.round(length * 0.9))].mean()
            r,g,b = B_npy[:, :, 0], B_npy[:, :, 1], B_npy[:, :, 2]
            value_B = (0.299*r+0.587*g+0.114*b) / 255.
            value_B = np.sort(value_B.flatten())
            length = value_B.shape[0]
            value_B = value_B[int(np.round(length * 0.1)) : int(np.round(length * 0.9))].mean()

            if int(np.round(value_A)) in self.low_range and int(np.round(value_B)) in self.high_range: break
            n_try += 1
        if n_try == self.N_TRY:
            self.A_paths[index % self.A_size]
            index = random.randint(0, self.__len__())
            return self.__getitem__(index)

        gray_mask = torch.ones(1, self.opt.fineSize, self.opt.fineSize) * value_A
        A_img_border = A_image.crop((x1-self.opt.fineSize//2, y1-self.opt.fineSize//2, x1+2*self.opt.fineSize, y1+2*self.opt.fineSize))
        A_Lab = torch.Tensor(color.rgb2lab(A_npy) / 100).permute([2, 0, 1])

        A_img = self.transform(A_img)
        A_img_border = self.transform(A_img_border)
        B_img = self.transform(B_img)

        if not 'images' in self.opt.name:
            mask = Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", "train", os.path.splitext(A_path.split("/")[-1])[0] + '_train_id.png'))
            # mask = Image.open(os.path.join("/ssd1/chenwy/bdd100k/seg/labels/", self.opt.phase, os.path.splitext(A_path.split("/")[-1])[0] + '_train_id.png'))
            mask = mask.crop((x1, y1, x1+self.opt.fineSize, y1+self.opt.fineSize)) # cropped mask for light_enhance_AB/seg
            mask = self._mask_transform(mask)
        else:
            mask = torch.zeros(1)
        
        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.

            r,g,b = A_img_border[0]+1, A_img_border[1]+1, A_img_border[2]+1
            A_gray_border = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray_border = torch.unsqueeze(A_gray_border, 0)
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path, 'mask': mask,
                'A_border': A_img_border, 'A_gray_border': A_gray_border,
                'A_Lab': A_Lab, 'gray_mask': gray_mask
                }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()