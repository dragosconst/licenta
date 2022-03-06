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

from Data_Processing.detection_dataset import PlayingCardsFRCNNDataset
from Utils.utils import intersection_over_union
from Utils.eval import eval
from references.engine import train_one_epoch, evaluate

def train_fccnn_reference(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader,
                          valid_dataloader: torch.utils.data.DataLoader,
                          lr_scheduler: torch.optim.lr_scheduler.MultiStepLR, device: str, num_epochs: int= 30) -> None:
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update learning rate
        lr_scheduler.step()
        # evaluate
        # evaluate(model, valid_dataloader, device=device)
        validate(model, valid_dataloader, device)

def train_frcnn(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,device: str,num_epochs: int =30) -> None:
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()   # train mode

        for images, targets in tqdm(train_dataloader):
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
        del loss_dict
        del loss_value
        del losses
        lr_scheduler.step()
        with torch.no_grad():
            validate(model, valid_dataloader, device)

@torch.inference_mode()
def validate(model: torch.nn.Module, valid_dataloader: torch.utils.data.DataLoader, device: str) -> None:
    model.eval()
    torch.cuda.empty_cache()

    true_positives = 0
    false_positives = 0
    total_len = 0
    for images, targets in tqdm(valid_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        """
        Detections is a list of dicts for each image, each image-dict being of the form
        d["boxes"], d["labels"] and d["scores"] with their respective meaning in the context of Faster R-CNN.
        """
        detections = model(images)
        # print(f"Detections are {detections}.")
        detections = [{k: v.to("cpu") for k, v in t.items()} for t in detections]
        # print(f"Targets are {targets}.")
        for detection, target in zip(detections, targets):
            tp, fp = eval(detection, target)
            total_len += len(target["boxes"])
            true_positives += tp
            false_positives += fp
    print(true_positives, false_positives)
    print(f"Precision is {true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0}.")
    print(f"Recall is {true_positives / total_len}.")
