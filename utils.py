import os
import torch
from model.vgg16 import Vgg16


class AvgLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_vgg16(model_folder):
    """Load the VGG16 model and initialize with pretrained weights"""
    if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
        vgg16 = Vgg16()
        model_path = os.path.join(model_folder, 'vgg16-397923af.pth')
        pretrained_dict = torch.load(model_path)
        model_dict = vgg16.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)

        vgg16.load_state_dict(model_dict)
        torch.save(vgg16.state_dict(), os.path.join(model_folder, 'vgg16.weight'))

