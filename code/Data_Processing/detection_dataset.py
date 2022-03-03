import os
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

"""
Dataset class for the Faster R-CNN network. The dataset itself contains both positive and negative examples, negative
examples simply lack an attached xml file.
"""
class PlayingCardsDataset(Dataset):

    """
    FP is expected to be an absolute filepath of the folder which contains the images.
    """
    def __init__(self, fp):
        self.fp = fp
        self.img_fps = sorted(os.listdir(fp))

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
                targets["boxes"].append([xmin, ymin, xmax, ymax])
                targets["labels"].append(name)
        return img, targets

