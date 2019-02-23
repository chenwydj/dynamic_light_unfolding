'''
https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
https://github.com/NVlabs/PL4NN/blob/master/src/loss.py
https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py

m = pytorch_msssim.MSSSIM()
img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))
print(pytorch_msssim.msssim(img1, img2))
print(m(img1, img2))
'''
    
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    return torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])

def uniform(window_size):
    return torch.Tensor([1.] * window_size)

def create_window(window_size, channel, sigma=5):
    _1D_window = gaussian(window_size, sigma)
    # _1D_window = uniform(window_size)
    _1D_window = _1D_window.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0) # add two more dimensions
    _2D_window /= _2D_window.sum()
    # (out_channels, in_channels / groups, height, width)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def l_cs(img1, img2, window, channel, padd = 0):

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    l = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    return l, cs


def _ssim(img1, img2, window, channel, size_average=True, full=False):
    l, cs = l_cs(img1, img2, window, channel)

    if full: return l.mean(), cs.mean()
    return (l * cs).mean()


def ssim(img1, img2, window_size=8, sigma=5, size_average=True, full=False):
    (_, channel, height, width) = img1.size()

    real_size = min(window_size, height, width)
    window = create_window(real_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return 1 - _ssim(img1, img2, window, channel, size_average, full=full)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=8, weights=None, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weights = weights
        if self.weights is not None:
            self.weights = F.conv2d(self.weights, self.window, padding=0, groups=self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
                if self.weights is not None: self.weights = self.weights.cuda(img1.get_device())
            window = window.type_as(img1)
            if self.weights is not None: self.weights = self.weights.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        l, cs = l_cs(img1, img2, self.window, channel)
        
        if self.weights is None: return 1 - (l * cs).mean()
        return ((1 - (l * cs)) * self.weights).mean()


def _msssim(img1, img2, window, channel, size_average=True):
    if img1.size() != img2.size():
        raise RuntimeError('Input images must have the same shape (%s vs. %s).' %
                           (img1.size(), img2.size()))
    if len(img1.size()) != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           len(img1.size()))

    # if type(img1) is not Variable or type(img2) is not Variable:
    #     raise RuntimeError('Input images must be Variables, not %s' % 
    #                         img1.__class__.__name__)

    num_scale = window.size(0)
    ms_l, ms_cs = [], []
    for i in range(num_scale):
        l, cs = l_cs(img1, img2, window[i], channel)
        ms_l.append(l.unsqueeze(0))
        ms_cs.append(cs.unsqueeze(0))
    ms_l = torch.cat(ms_l)
    ms_cs = torch.cat(ms_cs)
    # print("ms_l.size():", ms_l.size(), "ms_cs.size():", ms_cs.size())
    # print("prod of cs:", torch.prod(ms_cs, dim=0))
    # print("msssim:", (torch.prod(ms_cs, dim=0) * ms_l).mean())
    return (torch.prod(ms_cs, dim=0) * ms_l).mean()


def msssim(img1, img2, window_size=8, sigma=[0.5, 1., 2., 4., 8.], size_average=True):
    if img1.size() != img2.size():
        raise RuntimeError('Input images must have the same shape (%s vs. %s).' %
                           (img1.size(), img2.size()))
    if len(img1.size()) != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           len(img1.size()))

    # if type(img1) is not Variable or type(img2) is not Variable:
    #     raise RuntimeError('Input images must be Variables, not %s' % 
    #                         img1.__class__.__name__)

    (_, channel, height, width) = img1.size()
    real_size = min(window_size, height, width)
    
    window = []
    for i in range(len(self.sigmas)):
        window.append(create_window(real_size, channel, sigma=self.sigmas[i]).unsqueeze(0))
    window = torch.cat(self.window, dim=0)
    window = window.type_as(img1)
    
    return 1 - _msssim(img1, img2, window, channel, size_average=True)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=8, sigmas=[0.5, 1., 2., 4., 8.], size_average=True, channel=1):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.sigmas = sigmas
        self.window = []
        for i in range(len(self.sigmas)):
            self.window.append(create_window(window_size, self.channel, sigma=self.sigmas[i]).unsqueeze(0))
        self.window = torch.cat(self.window, dim=0)

    def forward(self, img1, img2):
        self.window = self.window.type_as(img1)
        return 1 - _msssim(img1, img2, self.window, self.channel, size_average=True)



class MSSSIM_L1(torch.nn.Module):
    def __init__(self, alpha = 0.84, window_size=8, sigmas=[0.5, 1., 2., 4., 8.], size_average=True, channel=1):
        super(MSSSIM_L1, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.alpha = alpha
        self.sigmas = sigmas
        self.window = []
        for i in range(len(self.sigmas)):
            self.window.append(create_window(window_size, self.channel, sigma=self.sigmas[i]).unsqueeze(0))
        self.window = torch.cat(self.window, dim=0)

    def forward(self, img1, img2):
        diff = torch.abs(img1 - img2)
        num_scale = self.window.size(0)
        l1 = 0
        for i in range(num_scale):
            l1 += F.conv2d(diff, self.window[i], padding=0, groups=channel).mean() # L1 loss weighted by Gaussian
        return (1 - self.alpha) * l1 + self.alpha * (1 - _msssim(img1, img2, self.window, self.channel, size_average=True))