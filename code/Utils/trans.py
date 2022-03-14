from typing import List, Optional, Dict, Tuple, Union
import random

from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch
import numpy as np
import cv2 as cv

class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for trans in self.transforms:
            img, target = trans(img, target)
        return img, target

class RandomAffineBoxSensitive():
    def __init__(self, degrees: Tuple[int, int]=(-1, 0), translate: Tuple[float, float]=(0.,0.)
                 ,scale: Union[float, Tuple[float, float]]=1., prob: float=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        h, w = image.shape[-2:]
        deg = torch.randint(*self.degrees, (1,)).item() if self.degrees[0] != -1 else 0
        trans = torch.rand(1).item()
        ta, tb = self.translate
        # ta = (1 - trans) * (-w * ta) + trans * (w * ta)
        # tb = (1 - trans) * (-h * tb) + trans * (h * tb)
        # trans = (ta, tb)

        # apply rotation
        image = F.rotate(image, deg)
        deg = np.deg2rad(-deg)
        center = (w//2, h//2)
        M = cv.getRotationMatrix2D(center, deg, 1.0)
        good_boxes = []
        good_indices = []
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            x3, y3, x4, y4 = x2, y1, x1, y2 # the other two points of the rectangle
            x1_ =  - (y1 - center[1]) * np.sin(deg) + (x1 - center[0]) * np.cos(deg) + center[0]
            y1_ = (y1 - center[1]) * np.cos(deg) + (x1 - center[0]) * np.sin(deg) + center[1]
            x2_ = - (y2 - center[1]) * np.sin(deg) + (x2 - center[0]) * np.cos(deg) + center[0]
            y2_ = (y2 - center[1]) * np.cos(deg) + (x2 - center[0]) * np.sin(deg) + center[1]
            x3_ = - (y3 - center[1]) * np.sin(deg) + (x3 - center[0]) * np.cos(deg) + center[0]
            y3_ = (y3 - center[1]) * np.cos(deg) + (x3 - center[0]) * np.sin(deg) + center[1]
            x4_ = - (y4 - center[1]) * np.sin(deg) + (x4 - center[0]) * np.cos(deg) + center[0]
            y4_ = (y4 - center[1]) * np.cos(deg) + (x4 - center[0]) * np.sin(deg) + center[1]
            new_box = (np.min((x1_, x2_, x3_, x4_)), np.min((y1_, y2_, y3_, y4_)),
                                    np.max((x1_, x2_, x3_, x4_)), np.max((y1_, y2_, y3_, y4_)))
            left, top, right, bot = new_box
            if left <= -10 or top <= -10 or right >= w + 10 or bot >= h + 10: # drop boxes that go out of bounds
                continue
            good_boxes.append(torch.as_tensor((left, top, right, bot)))
            good_indices.append(idx)
        target["boxes"] = good_boxes
        if "labels" in target:
            target["labels"] = target["labels"][good_indices]
        if "areas" in target:
            target["areas"] = target["areas"][good_indices]
        max_dx = float(ta * w)
        max_dy = float(tb * h)
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        trans = [tx, ty]

        # apply translation
        image = F.affine(img=image,angle=0, translate=trans, scale=1, shear=[0., 0.])
        good_boxes = []
        good_indices = []
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            left, top, right, bot = x1 + tx, y1 + ty, x2 + tx, y2 + ty
            if left <= -10 or top <= -10 or right >= w + 10 or bot >= h + 10: # drop boxes that go out of bounds
                continue
            good_boxes.append(torch.as_tensor((left, top, right, bot)))
            good_indices.append(idx)
        target["boxes"] = good_boxes
        if "labels" in target:
            target["labels"] = target["labels"][good_indices]
        if "areas" in target:
            target["areas"] = target["areas"][good_indices]

        # apply scaling
        if isinstance(self.scale, float):
            scale = self.scale
        else:
            scale_f = torch.rand(1).item()
            smin, smax = self.scale
            scale = (1 - scale_f) * smin + scale_f * smax
        image = F.affine(img=image, angle=0, translate=(0, 0), scale=scale, shear=[0., 0.])
        nh, nw = h * scale, w * scale
        good_indices = []
        good_boxes = []
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            left = scale * x1 + w//2 - nw//2
            top = scale * y1 + h//2 - nh//2
            right = scale * x2 + w//2 - nw//2
            bot = scale * y2 + h//2 - nh//2
            if left <= -10 or top <= -10 or right >= w + 10 or bot >= h + 10: # drop boxes that go out of bounds
                continue
            good_boxes.append(torch.as_tensor((left, top, right, bot)))
            good_indices.append(idx)
        target["boxes"] = good_boxes
        if "labels" in target:
            target["labels"] = target["labels"][good_indices]
        if "areas" in target:
            target["areas"] = target["areas"][good_indices]


        # area has obviously changed after the transforms, so we need to recalculate it
        if "area" in target:
            for idx, box in enumerate(target["boxes"]):
                x1, y1, x2, y2 = box
                target["area"][idx] = (x2 - x1) * (y2 - y1)
        return image, target

class RandomPerspectiveBoxSensitive():
    def __init__(self, dist_scale: float, prob: float=0.5):
        self.distortion_scale = dist_scale
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        height, width = image.shape[-2:]
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(self.distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(0, int(self.distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(self.distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(0, int(self.distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(self.distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(height - int(self.distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(self.distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(height - int(self.distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        fill = [float(0)] * F.get_image_num_channels(image)
        image = F.perspective(image, startpoints, endpoints, fill=fill)
        perspTransform = cv.getPerspectiveTransform(np.asarray(startpoints, dtype=np.float32), np.asarray(endpoints,
                                                                                                          dtype=np.float32))
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            x3, y3, x4, y4 = x2, y1, x1, y2 # the other two points of the rectangle
            new_box = cv.perspectiveTransform(np.asarray([((x1,y1),(x2,y2),(x3,y3),(x4,y4))]), perspTransform)
            (x1_, y1_), (x2_, y2_), (x3_, y3_), (x4_, y4_) = new_box[0]
            target["boxes"][idx] = torch.as_tensor((np.min((x1_, x2_, x3_, x4_)), np.min((y1_, y2_, y3_, y4_)),
                                    np.max((x1_, x2_, x3_, x4_)), np.max((y1_, y2_, y3_, y4_))))
            x1, y1, x2, y2 = target["boxes"][idx]
            if "area" in target:
                target["area"][idx] = (x2 - x1) * (y2 - y1)
        return image, target

class RandomColorJitterBoxSensitive():
    def __init__(self, brightness: float=0, contrast: float=0, saturation: float=0, hue: float=0,
                 prob: float=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        rgb_channels = image[:3, :, :]
        bright_factor = float(torch.empty(1).uniform_(max(0., 1 - self.brightness), self.brightness + 1))
        rgb_channels = F.adjust_brightness(rgb_channels, bright_factor)
        contr_factor = float(torch.empty(1).uniform_(max(0., 1 - self.contrast), self.contrast + 1))
        rgb_channels = F.adjust_contrast(rgb_channels, contr_factor)
        satur_fact = float(torch.empty(1).uniform_(max(0., 1 - self.saturation), self.saturation + 1))
        rgb_channels = F.adjust_saturation(rgb_channels, satur_fact)
        hue_fact = float(torch.empty(1).uniform_(-self.hue, self.hue))
        rgb_channels = F.adjust_hue(rgb_channels, hue_fact)
        image[:3, :, :] = rgb_channels

        return image, target

