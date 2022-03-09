from typing import List, Optional, Dict, Tuple
import random

from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch
import numpy as np
import cv2 as cv

class RandomAffineBoxSensitive():
    def __init__(self, degrees: Tuple[int, int], translate: Tuple[float, float]=(0.,0.)
                 ,prob: float=0.5):
        self.degrees = degrees
        self.translate = translate
        self.prob = prob

    def __call__(self, image, target):
        if random.random() > self.prob:
            return image, target
        h, w = image.shape[-2:]
        deg = torch.randint(*self.degrees, (1,)).item()
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
            target["boxes"][idx] = torch.Tensor((np.min((x1_, x2_, x3_, x4_)), np.min((y1_, y2_, y3_, y4_)),
                                    np.max((x1_, x2_, x3_, x4_)), np.max((y1_, y2_, y3_, y4_))))
            # target["boxes"][idx] =torch.Tensor( (x1_, y1_, x2_, y2_))
            # target["boxes"][idx] = F.rotate(box, deg)
        max_dx = float(ta * w)
        max_dy = float(tb * h)
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        trans = [tx, ty]

        # apply translation
        image = F.affine(img=image,angle=0, translate=trans, scale=1, shear=[0., 0.])
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            x1_ = min(max(x1 + ta, 0), w)
            y1_ = min(max(y1 + tb, 0), h)
            x2_ = min(max(x2 + ta, 0), w)
            y2_ = min(max(y2 + tb, 0), h)
            target["boxes"][idx] = torch.Tensor((x1_, y1_, x2_, y2_))

        # the boxes are most likely rhombuses by now - find the minimal rectangle that covers the bounding rhombuses
        # and recalculate the area
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
        perspTransform = cv.getPerspectiveTransform(startpoints, endpoints)
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            x3, y3, x4, y4 = x2, y1, x1, y2 # the other two points of the rectangle
            new_box = cv.perspectiveTransform((x1,y1,x2,y2,x3,y3,x4,y4), perspTransform)
            x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_ = new_box
            target["boxes"][idx] = (np.min((x1_, x2_, x3_, x4_)), np.min((y1_, y2_, y3_, y4_)),
                                    np.max((x1_, x2_, x3_, x4_)), np.max((y1_, y2_, y3_, y4_)))
            x1, y1, x2, y2 = target["boxes"][idx]
            target["area"][idx] = (x2 - x1) * (y2 - y1)
        return image, target
