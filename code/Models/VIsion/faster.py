from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
import math
import sys

import torch
import torch.utils.data
from torchvision.models import vgg19
from torchvision.ops import MultiScaleRoIAlign
from torch.nn import functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FasterRCNN

from Data_Processing.detection_dataset import PlayingCardsDataset
from Utils.utils import intersection_over_union
from Utils.eval import eval

"""
I will use a different RPN head for the Faster R-CNN network, for better fine tuning of the convolutional layer.
I will also use a different backbone, probably a pre-trained ImageNet network.
"""

# RPN head with a 5x5 kernel and custom mid channels size
class RPNHead_5(torch.nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_anchors: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=1, padding=1)
        self.cls = torch.nn.Conv2d(mid_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_regresor = torch.nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    def forward(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls = []
        bboxes = []
        for feature in x:
            h = F.relu(self.conv(feature))
            cls.append(self.cls(h))
            bboxes.append(self.bbox_regresor(h))
        return cls, bboxes

# Initialize a Faster R-CNN net with given params
def initialize_frcnn(backbone: torch.nn.Module, cls: int, anchor_gen: AnchorGenerator, roi_pooler: MultiScaleRoIAlign,
                     image_mean: float=None, image_std: float=None, min_size: int=800,
                     max_size: int=1333, rpn_head: torch.nn.Module=None) -> torch.nn.Module:
    if rpn_head is None:
        rpn_head = RPNHead(backbone.out_channels, anchor_gen.num_anchors_per_location()[0])

    return FasterRCNN(backbone=backbone, num_classes=cls, rpn_anchor_generator=anchor_gen, box_roi_pool=roi_pooler,
                      image_mean=image_mean, image_std=image_std, min_size=min_size, max_size=max_size,
                      rpn_head=rpn_head)


def train_frcnn(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,device: str,num_epochs: int =30) -> None:
    for epoch in tqdm(range(num_epochs)):
        model.train()   # train mode

        for images, targets in train_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) # pairwise sum of class and box regression losses for each detection for both rpn and Fast R-CNN

            loss_value = losses.item()

            losses.backward()
            optimizer.step()

            optimizer.zero_grad()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)
        print(f"Loss after epoc {epoch} is {loss_value}.")
        lr_scheduler.step()
        validate(model, valid_dataloader, device)

def validate(model: torch.nn.Module, valid_dataloader: torch.utils.data.DataLoader, device: str) -> None:
    model.eval()

    true_positives = 0
    false_positives = 0
    total_len = 0
    for images, targets in valid_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total_len += len(targets)

        """
        Detections is a list of dicts for each image, each image-dict being of the form
        d["boxes"], d["labels"] and d["scores"] with their respective meaning in the context of Faster R-CNN.
        """
        detections = model(images, targets)
        for detection, target in zip(detections, targets):
            tp, fp = eval(detection, target)
            true_positives += tp
            false_positives += fp
    print(f"Precision is {true_positives / (true_positives + false_positives)}.")
    print(f"Recall is {true_positives / total_len}.")
