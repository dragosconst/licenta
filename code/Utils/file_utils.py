from typing import List, Optional, Dict, Tuple
import glob
import os

import numpy as np
from PIL import Image
import torch

from Data_Processing.Raw_Train_Data.raw import parse_xml, possible_classes
from Utils.trans import RandomStretch

def get_image_files() -> List[str]:
    dir = "D:\\facultate stuff\\licenta\\data\\RAW\\dtd\\images"
    imgs_fns = []
    for subdirs in glob.glob(dir + "\\*"):
        for img_fn in glob.glob(os.path.join(subdirs, "*.jpg")):
            imgs_fns.append(img_fn)
    return imgs_fns

def get_random_bg_img(imgs_fns) -> Image.Image:
    idx = np.random.randint(0, len(imgs_fns))
    img = Image.open(imgs_fns[idx])
    return img


MAX_IM_SIZE = 350//1.75
def get_random_img(dir_path: str, stretched: bool=False) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    fps = glob.glob(os.path.join(dir_path, "*.jpg"))
    xmls = glob.glob(os.path.join(dir_path, "*.xml"))

    idx = np.random.randint(low=0, high=len(fps))
    img = Image.open(fps[idx])
    targets = parse_xml(xmls[idx])

    max_dim = np.max(img.size)
    w, h = img.size
    scale_factor = MAX_IM_SIZE / max_dim
    img = img.resize((int(w * scale_factor), int(h * scale_factor)))
    if stretched:
        img, targets = RandomStretch(sx=(0.6, 1.4), sy=(0.6, 1.4), prob=0.9)(img, targets)
    data_dict = {"boxes": [], "labels": []}
    for cls, *box in targets:
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_factor)
        y1 = int(y1 * scale_factor)
        x2 = int(x2 * scale_factor)
        y2 = int(y2 * scale_factor)
        data_dict["boxes"].append(torch.tensor((x1, y1, x2, y2)))
        data_dict["labels"].append(torch.tensor([possible_classes[cls]]))
    data_dict["boxes"] = torch.stack(data_dict["boxes"])
    data_dict["labels"] = torch.stack(data_dict["labels"])
    return img, data_dict

def load_img_and_xml(fp: str, im_name: str, stretched: bool=False) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    im_fp = os.path.join(fp, im_name + ".jpg")
    xml_fp = os.path.join(fp, im_name + ".xml")

    img = Image.open(im_fp)
    # img.show()
    targets = parse_xml(xml_fp)

    max_dim = np.max(img.size)
    w, h = img.size
    scale_factor = MAX_IM_SIZE / max_dim
    img = img.resize((int(w * scale_factor), int(h * scale_factor)))
    data_dict = {"boxes": [], "labels": []}
    for cls, *box in targets:
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_factor)
        y1 = int(y1 * scale_factor)
        x2 = int(x2 * scale_factor)
        y2 = int(y2 * scale_factor)
        data_dict["boxes"].append(torch.tensor((x1, y1, x2, y2)))
        data_dict["labels"].append(torch.tensor([possible_classes[cls]]))
    data_dict["boxes"] = torch.stack(data_dict["boxes"])
    data_dict["labels"] = torch.stack(data_dict["labels"])
    if stretched:
        img = torch.from_numpy(np.asarray(img, dtype=np.uint8))
        img = img.permute(2, 0, 1)
        img, data_dict = RandomStretch(sx=(0.6, 2.), sy=(0.6, 2.), prob=0.9)(img.to("cuda"), data_dict)
        img = img.permute(1, 2, 0).cpu()
        img = np.asarray(img, dtype=np.uint8)
        img = Image.fromarray(img)
    return img, data_dict
