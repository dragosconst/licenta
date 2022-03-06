import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

possible_classes = {"Ac": 1, "Ad": 2, "Ah": 3, "As": 4,
                    "2c": 5, "2d": 6, "2h": 7, "2s": 8,
                    "3c": 9, "3d": 10, "3h": 11, "3s": 12,
                    "4c": 13, "4d": 14, "4h": 15, "4s": 16,
                    "5c": 17, "5d": 18, "5h": 19, "5s": 20,
                    "6c": 21, "6d": 22, "6h": 23, "6s": 24,
                    "7c": 25, "7d": 26, "7h": 27, "7s": 28,
                    "8c": 29, "8d": 30, "8h": 31, "8s": 32,
                    "9c": 33, "9d": 34, "9h": 35, "9s": 36,
                    "10c": 37, "10d": 38, "10h": 39, "10s": 40,
                    "Jc": 41, "Jd": 42, "Jh": 43, "Js": 44,
                    "Qc": 45, "Qd": 46, "Qh": 47, "Qs": 48,
                    "Kc": 49, "Kd": 50, "Kh": 51, "Ks": 52,
                    "JOKER_red": 53, "JOKER_black": 54,
                    "not a card": 0
                    } # set of all possible classes
"""
Dataset class for the Faster R-CNN network. The dataset itself contains both positive and negative examples, negative
examples simply lack an attached xml file.
"""
class PlayingCardsFRCNNDataset(Dataset):

    """
    FP is expected to be an absolute filepath of the folder which contains the images.
    """
    def __init__(self, fp):
        self.fp = fp
        self.img_fps = sorted(glob.glob(os.path.join(fp, "*.jpg")))

    def __getitem__(self, idx):
        img = Image.open(self.img_fps[idx])
        targets = {"boxes": [], "labels": []}
        if os.path.exists(self.img_fps[idx][:-3] + "xml"):
            xml = ET.parse(self.img_fps[idx][:-3] + "xml")

            for obj in xml.findall('object'):
                name = obj.find('name').text

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                targets["boxes"].append(torch.Tensor([xmin, ymin, xmax, ymax]))
                targets["labels"].append(torch.Tensor([possible_classes[name]]).type(torch.int64))
        targets["boxes"] = torch.stack(targets["boxes"])
        targets["labels"] = torch.stack(targets["labels"]).squeeze(dim=1)
        targets["image_id"] = torch.Tensor([idx])
        targets["area"] = (xmax - xmin) * (ymax - ymin)
        targets["iscrowd"] = False
        img = transforms.PILToTensor()(img)
        img = img[:3, :, :] # drop alpha channel, if it exists
        return transforms.ConvertImageDtype(torch.float)(img), targets

    def __len__(self):
        return len(self.img_fps)