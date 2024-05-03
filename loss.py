import os

import torch
from torch import nn
import torch.nn.functional as F
from model.vgg16 import Vgg16
from utils import init_vgg16


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class Laplacian_Loss(nn.Module):
    def __init__(self):
        super(Laplacian_Loss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class VGG_loss(nn.Module):
    def __init__(self, vgg_model='./model/'):
        super(VGG_loss, self).__init__()
        self.vgg = Vgg16()
        init_vgg16(vgg_model)
        self.vgg.load_state_dict(torch.load(os.path.join(vgg_model, "vgg16.weight")))
        self.vgg.cuda()
        self.L1 = nn.L1Loss()

    def forward(self, depth, depth_pred):
        depth_vgg = self.vgg(depth)
        depth_1 = depth_vgg[0].detach().clone().requires_grad_(False)
        depth_2 = depth_vgg[1].detach().clone().requires_grad_(False)
        depth_pred_1, depth_pred_2 = self.vgg(depth_pred)
        loss = self.L1(depth_pred_1, depth_1) + self.L1(depth_pred_2, depth_2)
        return loss
