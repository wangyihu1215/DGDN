import random
from PIL import Image
import torchvision.transforms.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, original, haze, depth):
        assert original.size == haze.size
        assert original.size == depth.size
        for t in self.transforms:
            original, haze, depth = t(original, haze, depth)
        return original, haze, depth


class Resize(object):
    def __init__(self, size):
        self.size = tuple(size)  # size(w, h)

    def __call__(self, original, haze, depth):
        assert original.size == haze.size
        assert original.size == depth.size

        original = original.resize(self.size, Image.BILINEAR)
        haze = haze.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)
        return original, haze, depth


class RandomHorizontallyFlip(object):
    def __call__(self, original, haze, depth):
        if random.random() < 0.5:
            original = original.transpose(Image.FLIP_LEFT_RIGHT)
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            return original, haze, depth
        else:
            return original, haze, depth


class ToTensor(object):
    def __call__(self, original, haze, depth):
        # PIL -> Tensor
        original = F.to_tensor(original)
        haze = F.to_tensor(haze)
        depth = F.to_tensor(depth)
        return original, haze, depth
