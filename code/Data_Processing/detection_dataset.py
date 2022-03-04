import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

from Data_Processing.Raw_Train_Data.raw import possible_classes

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
        img = transforms.PILToTensor()(img)
        img = img[:3, :, :] # drop alpha channel, if it exists
        return transforms.ConvertImageDtype(torch.float)(img), targets

    def __len__(self):
        return len(self.img_fps)

