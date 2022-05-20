from typing import List, Optional, Dict, Tuple, Union
import random

from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch
import numpy as np
import cv2 as cv


class MyCompose:
    def __init__(self, transforms):
        self.transforms = transforms
        self.debug_mode = transforms[0].debug
        for transform in transforms:
            assert transform.debug == self.debug_mode

    def __call__(self, img, target):
        if self.debug_mode:
            debug = []
            for box in target["boxes"]:
                x1, y1, x2, y2 = box
                debug.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        for trans in self.transforms:
            if self.debug_mode:
                img, target, debug = trans(img, target, debug)
            else:
                img, target = trans(img, target)
        if self.debug_mode:
            return img, target, debug
        return img, target


class RandomAffineBoxSensitive:
    def __init__(self, degrees: Tuple[int, int]=(-1, 0), translate: Tuple[float, float]=(0.,0.)
                 ,scale: Union[float, Tuple[float, float]]=1., prob: float=0.5, debug: bool=False):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.prob = prob
        self.debug = debug

    def rotate_points_around_center(self, theta: float, points: List[Tuple[float, float]], center: Tuple[float, float])\
        -> List[Tuple[float, float]]:
        """
        Apply rotation of theta angle to a list of points, centered around the center parameter.
        :param theta: rotation angle in radians
        :param points: list of (x,y) to be transformed
        :param center: rotation center
        :return: list of (x',y') after transformation
        """
        cx, cy = center
        result = []
        for x, y in points:
            xprime = - (y - cy) * np.sin(theta) + (x - cx) * np.cos(theta) + cx
            yprime = (y - cy) * np.cos(theta) + (x - cx) * np.sin(theta) + cy
            result.append((xprime, yprime))
        return result

    def __call__(self, image, target, debug: List=None):
        if random.random() > self.prob:
            if self.debug:
                if debug is not None:
                    return image, target, debug
                debug_boxes = []
                for box in target["boxes"]:
                    x1, y1, x2, y2 = box
                    debug_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                return image, target, debug_boxes
            return image, target
        h, w = image.shape[-2:]
        deg = torch.randint(*self.degrees, (1,)).item() if self.degrees[0] != -1 else 0
        trans = torch.rand(1).item()
        ta, tb = self.translate

        # apply rotation
        image = F.rotate(image, deg)
        deg = np.deg2rad(-deg)
        center = (w//2, h//2)
        M = cv.getRotationMatrix2D(center, deg, 1.0)
        good_boxes = []
        good_indices = []
        if self.debug:
            debug_boxes = []
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            x3, y3, x4, y4 = x2, y1, x1, y2 # the other two points of the rectangle
            (x1_, y1_), (x2_, y2_), (x3_, y3_), (x4_, y4_) = self.rotate_points_around_center(deg, [(x1, y1), (x2, y2),
                                                                                                    (x3, y3), (x4, y4)],
                                                                                              center)
            new_box = (np.min((x1_, x2_, x3_, x4_)), np.min((y1_, y2_, y3_, y4_)),
                                    np.max((x1_, x2_, x3_, x4_)), np.max((y1_, y2_, y3_, y4_)))
            left, top, right, bot = new_box
            if left <= -10 or top <= -10 or right >= w + 10 or bot >= h + 10: # drop boxes that go out of bounds
                continue
            if self.debug:
                if debug is None:
                    debug_boxes.append([(x1_, y1_), (x3_, y3_), (x2_, y2_), (x4_, y4)])
                else:
                    points = debug[idx]
                    points = self.rotate_points_around_center(deg, points, center)
                    debug_boxes.append(points)
            good_boxes.append(torch.as_tensor((left, top, right, bot)))
            good_indices.append(idx)
        target["boxes"] = good_boxes
        if "labels" in target:
            target["labels"] = target["labels"][good_indices]
        if "area" in target:
            target["area"] = target["area"][good_indices]
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
            if self.debug:
                points = debug_boxes[idx]
                points = [(x + tx, y + ty) for x, y in points]
                debug_boxes[idx] = points
            good_boxes.append(torch.as_tensor((left, top, right, bot)))
            good_indices.append(idx)
        target["boxes"] = good_boxes
        if "labels" in target:
            target["labels"] = target["labels"][good_indices]
        if "area" in target:
            target["area"] = target["area"][good_indices]

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

            if self.debug:
                points = debug_boxes[idx]
                points = [(x * scale + w//2 - nw//2, y * scale + h//2 - nh//2) for x, y in points]
                debug_boxes[idx] = points
            good_boxes.append(torch.as_tensor((left, top, right, bot)))
            good_indices.append(idx)
        target["boxes"] = good_boxes
        if type(target["boxes"]) == list and len(target["boxes"]) > 0:
            target["boxes"] = torch.stack(target["boxes"])
        elif len(target["boxes"]) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        if "labels" in target:
            target["labels"] = target["labels"][good_indices]
        if "area" in target:
            target["area"] = target["area"][good_indices]

        # area has obviously changed after the transforms, so we need to recalculate it
        if "area" in target:
            for idx, box in enumerate(target["boxes"]):
                x1, y1, x2, y2 = box
                target["area"][idx] = (x2 - x1) * (y2 - y1)
        if self.debug:
            return image, target, debug_boxes
        return image, target


class RandomPerspectiveBoxSensitive:
    def __init__(self, dist_scale: float, prob: float=0.5, debug: bool=False):
        self.distortion_scale = dist_scale
        self.prob = prob
        self.debug = debug

    def __call__(self, image, target, debug: List=None):
        if random.random() > self.prob:
            if self.debug:
                if debug is not None:
                    return image, target, debug
                debug_boxes = []
                for box in target["boxes"]:
                    x1, y1, x2, y2 = box
                    debug_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                return image, target, debug_boxes
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
        if self.debug:
            debug_boxes = []
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            x3, y3, x4, y4 = x2, y1, x1, y2 # the other two points of the rectangle
            new_box = cv.perspectiveTransform(np.asarray([((x1,y1),(x2,y2),(x3,y3),(x4,y4))]), perspTransform)
            (x1_, y1_), (x2_, y2_), (x3_, y3_), (x4_, y4_) = new_box[0]
            target["boxes"][idx] = torch.as_tensor((np.min((x1_, x2_, x3_, x4_)), np.min((y1_, y2_, y3_, y4_)),
                                    np.max((x1_, x2_, x3_, x4_)), np.max((y1_, y2_, y3_, y4_))))
            if self.debug:
                if debug is None:
                    debug_boxes.append([(x1_, y1_), (x3_, y3_), (x2_, y2_), (x4_, y4_)])
                else:
                    points = debug[idx]
                    points = cv.perspectiveTransform(np.asarray([points]), perspTransform)
                    debug_boxes.append(points)
            x1, y1, x2, y2 = target["boxes"][idx]
            if "area" in target:
                target["area"][idx] = (x2 - x1) * (y2 - y1)
        if self.debug:
            return image, target, debug_boxes
        return image, target


class RandomColorJitterBoxSensitive:
    def __init__(self, brightness: float=0, contrast: float=0, saturation: float=0, hue: float=0,
                 prob: float=0.5, debug: bool=False):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
        self.debug = debug

    def __call__(self, image, target, debug: List=None):
        if random.random() > self.prob:
            if self.debug:
                if debug is not None:
                    return image, target, debug
                debug_boxes = []
                for box in target["boxes"]:
                    x1, y1, x2, y2 = box
                    debug_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                return image, target, debug_boxes
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

        if self.debug:
            debug_boxes = []
            for box in target["boxes"]:
                x1, y1, x2, y2 = box
                debug_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            return image, target, debug_boxes
        return image, target


class RandomStretch:
    def __init__(self, sx: Tuple[float, float], sy: Tuple[float, float], prob: float=0.5):
        self.sx = sx
        self.sy = sy
        self.prob = prob
        self.debug = False

    def __call__(self, image, target, debug: List=None):
        if random.random() > self.prob:
            if self.debug:
                if debug is not None:
                    return image, target, debug
                debug_boxes = []
                for box in target["boxes"]:
                    x1, y1, x2, y2 = box
                    debug_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                return image, target, debug_boxes
            return image, target
        if self.debug:
            debug_boxes = []

        *_, h, w = image.shape
        if random.random() <= 0.5:
            smin, smax = self.sx
            scale_f = torch.rand(1).item()
            scalex = (1 - scale_f) * smin + scale_f * smax
            scaley = 1
            image = F.resize(img=image, size=[int(h), int(w * scalex)])
            nh, nw = h, w * scalex
        else:
            smin, smax = self.sy
            scale_f = torch.rand(1).item()
            scalex = 1
            scaley = (1 - scale_f) * smin + scale_f * smax
            image = F.resize(img=image, size=[int(h * scaley), int(w)])
            nh, nw = h * scaley, w
        good_indices = []
        good_boxes = []
        for idx, box in enumerate(target["boxes"]):
            x1, y1, x2, y2 = box
            left = x1 * scalex
            top = y1 * scaley
            right = x2 * scalex
            bot = y2 * scaley


            if self.debug:
                if debug is None:
                    debug_boxes.append([(left, top), (right, top), (right, bot), (left, bot)])
                else:
                    points = debug[idx]
                    # points = cv.perspectiveTransform(np.asarray([points]), perspTransform)
                    debug_boxes.append(points)
            good_boxes.append(torch.as_tensor((left, top, right, bot)))
            good_indices.append(idx)
        target["boxes"] = good_boxes
        if type(target["boxes"]) == list and len(target["boxes"]) > 0:
            target["boxes"] = torch.stack(target["boxes"])
        elif len(target["boxes"]) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        if "labels" in target:
            target["labels"] = target["labels"][good_indices]
        if "area" in target:
            target["area"] = target["area"][good_indices]
        if self.debug:
            return image, target, debug_boxes
        return image, target


class RandomGaussianNoise:
    def __init__(self, mean: float=0, var: float=1, prob: float=0.5, debug: bool=False):
        self.mean = mean
        self.var = var
        self.prob = prob
        self.debug = debug

    def __call__(self, image, target, debug: List=None):
        if random.random() > self.prob:
            if self.debug:
                if debug is not None:
                    return image, target, debug
                debug_boxes = []
                for box in target["boxes"]:
                    x1, y1, x2, y2 = box
                    debug_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                return image, target, debug_boxes
            return image, target
        rgb_channels = image[:3, :, :].cpu()
        rgb_channels = T.ConvertImageDtype(torch.float32)(rgb_channels)
        rgb_channels = rgb_channels + torch.randn(rgb_channels.size()) * self.var + self.mean
        image[:3, :, :] = T.ConvertImageDtype(image.dtype)(rgb_channels)
        if self.debug:
            debug_boxes = []
            for box in target["boxes"]:
                x1, y1, x2, y2 = box
                debug_boxes.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            return image, target, debug_boxes
        return image, target
