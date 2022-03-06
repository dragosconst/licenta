from typing import List, Optional, Dict, Tuple

import torch
import torch.utils.data
from torchvision.models import vgg19
from torch.nn import functional as F
from torchvision.models.detection.rpn import AnchorGenerator

from Data_Processing.detection_dataset import PlayingCardsFRCNNDataset

"""
Batches will be lists of (image, targets) tuples, this changes the whole batch into the tuple
(img1, img2, ...), (target1, target2, ...).
We must do this because this is the format the pytorch Faster R-CNN implementation expects.
"""
def collate_fn(batch):
    return tuple(zip(*batch))

def load_dataloader(batch_size: int = 16, shuffle: bool = True) -> Tuple[PlayingCardsFRCNNDataset, torch.utils.data.DataLoader]:
    dataset = PlayingCardsFRCNNDataset("D:\\facultate stuff\\licenta\\data\\train_imgs_full\\")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return dataset, dataloader

def get_loader(dataset: torch.utils.data.Dataset, batch_size: int = 16, shuffle: bool = True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)


    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou
