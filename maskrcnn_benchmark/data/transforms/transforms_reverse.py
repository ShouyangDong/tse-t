# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import copy
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        restore_list = []
        for t in self.transforms:
            image, target,restore_params = t(image, target)
            restore_list.append(restore_params)
        return image, [target,restore_list]

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        sz_origin = copy.copy(image.size)
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target,sz_origin


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        trans_used = False
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
            trans_used = True
        return image, target,trans_used

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        trans_used = False
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
            trans_used = True

        return image, target,trans_used

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target,None


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target,None


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target,None


class Resize_Reverse(object):
    def __call__(self,target,origin_sz):
        target = target.resize(origin_sz)
        return target  

def index_transpose(idx,shape_wh,dim,anchor=9):
    W,H = int(shape_wh[0]),int(shape_wh[1])
    ind_y,ind_x,ind_a = np.unravel_index(idx,[H,W,anchor])
    if dim == FLIP_LEFT_RIGHT:
        ind_x = int(W) - 1 - ind_x
    elif dim == FLIP_TOP_BOTTOM:
        ind_y = int(H) - 1 - ind_y
    else:
        raise('unknow transpose shape') 
    return np.ravel_multi_index(np.vstack([ind_y,ind_x,ind_a]),[H,W,anchor])

class RandomHorizontalFlip_Reverse(object):
    def __call__(self, target,hor_fliped):
        if hor_fliped:
            target = target.transpose(FLIP_LEFT_RIGHT)
            target.extra_fields['sel_ind'] = torch.tensor(index_transpose(target.extra_fields['sel_ind'].cpu(),target.get_field('feat_w_h'),dim=FLIP_LEFT_RIGHT))
        return target


class RandomVerticalFlip_Reverse(object):
    def __call__(self,target,ver_fliped):
        if ver_fliped:
            target = target.transpose(FLIP_TOP_BOTTOM)
            target.extra_fields['sel_ind'] = torch.tensor(index_transpose(target.extra_fields['sel_ind'].cpu(),target.get_field('feat_w_h'),dim=FLIP_TOP_BOTTOM))
        return target

class ColorJitter_Reverse(object):
     def __call__(self,target,par):
        return target

class ToTensor_Reverse(object):
     def __call__(self,target,par):
            return target


class Normalize_Reverse(object):
     def __call__(self,target,par):
            return target

#./maskrcnn_benchmark/data/transforms/build.py:40
def trans_reverse(target,reverse_info):
    target_out = copy.deepcopy(target)
    if target.bbox.shape[0] == 0:
        target_out.size = reverse_info[1]
        return target_out
    transform = [
            ColorJitter_Reverse(),
            Resize_Reverse(),
            RandomHorizontalFlip_Reverse(),
            RandomVerticalFlip_Reverse(),
            ToTensor_Reverse(),
            Normalize_Reverse(),
        ]
    for _fn,_para in zip(transform,reverse_info):
        target_out = _fn(target_out,_para)
    return target_out
    