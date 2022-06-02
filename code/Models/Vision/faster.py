from typing import List, Optional, Dict, Tuple
import math
import sys
# sys.path.append("D:\facultate stuff\licenta\code")
import time
import copy
import random
from collections import deque

from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
import torch
import torch.utils.data
import torchvision
from torchvision.models import vgg19
from torchvision.ops import MultiScaleRoIAlign
from torch.optim import Adam, SGD
from torch.nn import functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_bounding_boxes


from Data_Processing.detection_dataset import PlayingCardsFRCNNDataset
from Utils.utils import intersection_over_union
from Utils.eval import eval
from references.engine import train_one_epoch, evaluate
from Utils.utils import load_dataloader, get_loader, load_negative_dataloader
from Utils.trans import RandomAffineBoxSensitive, RandomPerspectiveBoxSensitive
from Data_Processing.detection_processing_resnet import second_nms, filter_under_thresh
from Image_Processing.detection_draw import draw_detection
import Utils.trans as T


def train_frcnn_reference(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader,
                          valid_dataloader: torch.utils.data.DataLoader,
                          lr_scheduler: torch.optim.lr_scheduler.MultiStepLR, device: str, num_epochs: int= 30,
                          start_from: int=None) -> None:
    # gradient clipping
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10, accumulate=16)
        # torch.save(frcnn.state_dict(), "D:\\facultate stuff\\licenta\\data\\frcnn_resnet50_5k_per_class_slices_e" + str(epoch +
        #                                                                     (0 if start_from is None else start_from)) +
        #     ".pt")
        torch.save(frcnn.state_dict(), "/mnt/d/facultate stuff/licenta/data/frcnn_resnet50_5k_per_class_stretchedref_e" + str(epoch +
                                                                            (0 if start_from is None else start_from)) +
            ".pt")

        # update learning rate
        lr_scheduler.step()
        # evaluate
        # evaluate(model, valid_dataloader, device=device)
        validate(model, valid_dataloader, device)


def train_frcnn(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,device: str,num_epochs: int =30, start_from: int=0) -> None:
    # gradient clipping
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    accumulate = 22
    print_freq = 20

    moving_loss = deque(maxlen=20)
    moving_class = deque(maxlen=20)
    moving_obj = deque(maxlen=20)
    moving_box_reg = deque(maxlen=20)
    moving_rpn_box_reg = deque(maxlen=20)
    for epoch in range(num_epochs):
        model.train()   # train mode

        full_loss = []
        optimizer.zero_grad()
        for idx, (images, targets) in enumerate(tqdm(train_dataloader)):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) # pairwise sum of class and box regression losses for each detection for both rpn and Fast R-CNN

            loss_value = losses.item()
            full_loss.append(loss_value)

            losses = losses / accumulate  # normalize losses
            losses.backward()

            moving_loss.append(loss_value)
            moving_class.append(loss_dict['loss_classifier'])
            moving_box_reg.append(loss_dict['loss_box_reg'])
            moving_obj.append(loss_dict['loss_objectness'])
            moving_rpn_box_reg.append(loss_dict['loss_rpn_box_reg'])
            if idx % print_freq == 0:
                print(f"Lr is {lr_scheduler.get_lr()}.")
                print(f"Loss at step [{idx}\\{len(train_dataloader)}] is: {sum(moving_loss)/len(moving_loss):.4f}, loss_classifier: {sum(moving_class)/len(moving_class):.4f},"
                      f" loss_box_reg:{sum(moving_box_reg)/len(moving_box_reg):.4f}, loss_objectness: {sum(moving_obj)/len(moving_obj):.4f},"
                      f" loss_rpn_box_reg:{sum(moving_rpn_box_reg)/len(moving_rpn_box_reg):.4f}")
            if (idx + 1) % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)
        print(f"Loss after epoc {epoch} is {sum(full_loss)/len(full_loss)}.")
        validate(model, valid_dataloader, device)
        torch.save(frcnn.state_dict(), "/mnt/d/facultate stuff/licenta/data/frcnn_resnet50_betterrpn_e" + str(epoch +
                                                                            (0 if start_from is None else start_from)) +
            ".pt")

        if lr_scheduler is not None:
            lr_scheduler.step()


def initialize_frcnn(backbone: torch.nn.Module, cls: int, anchor_gen: AnchorGenerator=None, roi_pooler: MultiScaleRoIAlign=None,
                     image_mean: float=None, image_std: float=None, min_size: int=800,
                     max_size: int=1333, rpn_head: torch.nn.Module=None) -> torch.nn.Module:
    # if rpn_head is None:
    #     rpn_head = RPNHead(backbone.out_channels, anchor_gen.num_anchors_per_location()[0])

    return FasterRCNN(backbone=backbone, num_classes=cls, rpn_anchor_generator=anchor_gen, box_roi_pool=roi_pooler,
                      image_mean=image_mean, image_std=image_std, min_size=min_size, max_size=max_size,
                      rpn_head=rpn_head)


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
            if len(detection["boxes"]) > 0:
                filter_under_thresh(detection)
                second_nms(detection)
            tp, fp = eval(detection, target)
            total_len += len(target["boxes"])
            true_positives += tp
            false_positives += fp
    print(true_positives, false_positives)
    print(f"Precision is {true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0}.")
    print(f"Recall is {true_positives / total_len}.")


def get_faster(model_path: str, mobilenet: bool=False) -> torch.nn.Module:
    if not mobilenet:
        frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to("cuda")
    else:
        frcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True).to("cuda")
    # frcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to("cuda")  # low res
    in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 55).to("cuda")
    frcnn.load_state_dict(torch.load(model_path))
    # frcnn.eval()
    frcnn.to("cuda")
    return frcnn


# when running this script, train various nets
if __name__ == "__main__":
    # torch.cuda.empty_cache()
    frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=4).to("cuda")
    # frcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, trainable_backbone_layers=5).to("cuda")
    # frcnn = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to("cuda")  # low res
    in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 55).to("cuda")
    # frcnn.backbone = backbone


    # TODO: filter out the bad labels from the gigantic dataset
    dataset, dataloader = load_dataloader(batch_size=1)
    dataset_test = copy.deepcopy(dataset)
    targets = dataset.targets()
    indices = np.asarray([x for x in range(len(dataset))])
    indices = indices[..., np.newaxis]
    train_idx, _, val_idx, _ = iterative_train_test_split(indices, np.asarray(targets), test_size=0.2)
    # indices = torch.randperm(len(dataset)).tolist()
    random.seed(1)
    torch.manual_seed(1)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset_test, val_idx)
    # train_set = torch.utils.data.Subset(dataset, indices)
    # val_set = torch.utils.data.Subset(dataset, indices)
    # train_set.dataset.transforms = T.MyCompose((RandomAffineBoxSensitive(degrees=(0, 45), prob=0.4),
    #                                             RandomPerspectiveBoxSensitive(dist_scale=0.3, prob=0.2)))

    train_loader = get_loader(train_set, batch_size=3, shuffle=True, num_workers=8)
    valid_loader = get_loader(val_set, batch_size=3, shuffle=False, num_workers=8)

    # sgd = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # train_frcnn(frcnn, adam, lr_scheduler=lr_sched, train_dataloader=train_loader, valid_dataloader=valid_loader,
    #             device="cuda", num_epochs=30)

    # frcnn.load_state_dict(torch.load("D:\\facultate stuff\\licenta\\data\\frcnn_resnet50_5k_per_class_slices.pt"))
    frcnn.load_state_dict(torch.load("/mnt/d/facultate stuff/licenta/data/frcnn_resnet50_betterrpn_e0.pt"))
    # frcnn.eval()

    frcnn.to("cuda")
    params = [p for p in frcnn.parameters() if p.requires_grad]
    adam = Adam(params, lr=1e-3)
    # lr_sched = torch.optim.lr_scheduler.MultiStepLR(adam, [10, 20, 25], gamma=0.1)
    lr_sched = torch.optim.lr_scheduler.StepLR(adam, step_size=4, gamma=0.1)
    # train_set, train_loader = load_negative_dataloader(batch_size=2, shuffle=True)
    # validate(frcnn, valid_loader, "cuda")

    train_frcnn(frcnn, adam, lr_scheduler=lr_sched, train_dataloader=train_loader, valid_dataloader=valid_loader,
                device="cuda", num_epochs=7, start_from=1)
    # torch.save(frcnn.state_dict(), "D:\\facultate stuff\\licenta\\data\\frcnn_resnet50_5k_per_class_slices.pt")
    torch.save(frcnn.state_dict(), "/mnt/d/facultate stuff/licenta/data/frcnn_resnet50_betterrpn.pt")
    # torch.save(frcnn.state_dict(), "D:\\facultate stuff\\licenta\\data\\mobilenet_v3_320_large.pt")
    # torch.save(frcnn.state_dict(), "D:\\facultate stuff\\licenta\\data\\frcnn_custom.pt")
