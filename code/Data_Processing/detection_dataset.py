import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms import functional as F
import numpy as np

from Data_Processing.Raw_Train_Data.raw import possible_classes

"""
Dataset class for the Faster R-CNN network. The dataset itself contains both positive and negative examples, negative
examples simply lack an attached xml file.
"""
"""
Dataset class for the Faster R-CNN network. The dataset itself contains both positive and negative examples, negative
examples simply lack an attached xml file.
"""
class PlayingCardsFRCNNDataset(Dataset):

    """
    FP is expected to be an absolute filepath of the folder which contains the images.
    """
    def __init__(self, fp, transforms=None):
        self.fp = fp
        self.transforms = transforms
        self.img_fps = sorted(glob.glob(os.path.join(fp, "*.jpg")))

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx[0]
        img = Image.open(self.img_fps[idx]).convert("RGB")
        targets = {"boxes": [], "labels": [], "image_id": [], "area": [], "iscrowd": []}
        if os.path.exists(self.img_fps[idx][:-3] + "xml"):
            xml = ET.parse(self.img_fps[idx][:-3] + "xml")

            for obj in xml.findall('object'):
                name = obj.find('name').text

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                targets["boxes"].append([xmin, ymin, xmax, ymax])
                targets["labels"].append(possible_classes[name])
                targets["area"].append((xmax - xmin) * (ymax - ymin))
                targets["iscrowd"].append(0)
        targets["boxes"] = torch.as_tensor(targets["boxes"], dtype=torch.float32)
        targets["labels"] = torch.as_tensor(targets["labels"], dtype=torch.int64)
        targets["image_id"] = torch.Tensor([idx]).type(torch.int64)
        targets["area"] = torch.as_tensor(targets["area"], dtype=torch.float32)
        targets["iscrowd"] = torch.as_tensor(targets["iscrowd"])
        img = transforms.PILToTensor()(img)
        img = img[:3, :, :] # drop alpha channel, if it exists
        img = transforms.ToPILImage()(img)

        img = F.to_tensor(img)
        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        return img, targets

    def __len__(self):
        return len(self.img_fps)

    # returns all targets
    def targets(self):
        targets = []
        for idx in range(len(self.img_fps)):
            if os.path.exists(self.img_fps[idx][:-3] + "xml"):
                xml = ET.parse(self.img_fps[idx][:-3] + "xml")
                cr_target = [0] * len(possible_classes)
                for obj in xml.findall('object'):
                    name = obj.find('name').text
                    cr_target[possible_classes[name]] = 1
                targets.append(cr_target)
        return targets