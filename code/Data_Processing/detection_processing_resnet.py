from typing import List, Optional, Dict, Tuple

import torch

from Utils.utils import intersection_over_union as iou
from Data_Processing.Raw_Train_Data.raw import pos_cls_inverse, possible_classes
"""
There are a couple problems with my ResNet50 network so far. Even the one that used on-line augmentation has a LOT of
false positives, but, fortunately enough, they almost always overlap true positives and almost always have a lower score
than a false positive. Therefore, I will apply a second layer of NMS over the results, this time supressing ANY detection
that overlaps a detection with a greater score, and also suppresses detections with under 0.3 (or perhaps another constant)
score. The second condition comes in handy with false positives that are not even close to being cards, they very rarely do
happen, but have almost negligible scores, usually.
"""

# in-place operation
def second_nms(detection: Dict[str, torch.Tensor]) -> None:
    boxes = detection["boxes"]
    bad_boxes = set()
    scores = detection["scores"]
    labels = detection["labels"]
    # the boxes are already sorted according to scores
    for idx, box in enumerate(boxes):
        for box2 in boxes[idx+1:]:
            if iou(box, box2) >= 0.3:
                bad_boxes.add(str(box2))
    good_boxes = []
    good_scores = []
    good_labels = []
    for box, score, label in zip(boxes, scores, labels):
        if str(box) not in bad_boxes:
            good_boxes.append(box)
            good_scores.append(score)
            good_labels.append(label)
    if len(good_boxes) > 0:
        detection["boxes"] = torch.stack(good_boxes)
        detection["scores"] = torch.stack(good_scores)
        detection["labels"] = torch.stack(good_labels)
    else:
        detection["boxes"] = torch.as_tensor([])
        detection["scores"] = torch.as_tensor([])
        detection["labels"] = torch.as_tensor([])

# in-place operation
def filter_under_thresh(detection: Dict[str, torch.Tensor]) -> None:
    boxes = detection["boxes"]
    bad_boxes = set()
    scores = detection["scores"]
    labels = detection["labels"]
    # the boxes are already sorted according to scores
    for idx, score in enumerate(scores):
        if score < 0.75:
            bad_boxes.add(str(boxes[idx])) # use boxes coordinates because they are the only unique property of bounding boxes

    good_boxes = []
    good_scores = []
    good_labels = []
    for box, score, label in zip(boxes, scores, labels):
        if str(box) not in bad_boxes:
            good_boxes.append(box)
            good_scores.append(score)
            good_labels.append(label)
    if len(good_boxes) > 0:
        detection["boxes"] = torch.stack(good_boxes)
        detection["scores"] = torch.stack(good_scores)
        detection["labels"] = torch.stack(good_labels)
    else:
        detection["boxes"] = torch.as_tensor([])
        detection["scores"] = torch.as_tensor([])
        detection["labels"] = torch.as_tensor([])


def filter_detections_by_game(game: str, detection: Dict[str, torch.Tensor]) -> None:
    """
    In-place operation that removes impossible detections (i.e. Joker in a game that doesn't utilize Jokers) from
    detections. Useful for removing faulty joker detections and removing irrelevant card detections
    :param game: string repesenting game name
    :param detection: detection given by model
    :return:
    """
    bad_labels = []
    if game == "Blackjack" or game == "Razboi" or game == "Poker Texas Hold'Em":
        bad_labels = [possible_classes["JOKER_black"], possible_classes["JOKER_red"]]
    elif game == "Septica":
        bad_labels = [possible_classes["2c"], possible_classes["2d"], possible_classes["2h"], possible_classes["2s"],
                      possible_classes["3c"], possible_classes["3d"], possible_classes["2h"], possible_classes["3s"],
                      possible_classes["4c"], possible_classes["4d"], possible_classes["4h"], possible_classes["4s"],
                      possible_classes["5c"], possible_classes["5d"], possible_classes["5h"], possible_classes["5s"],
                      possible_classes["6c"], possible_classes["6d"], possible_classes["6h"], possible_classes["6s"],
                      possible_classes["JOKER_black"], possible_classes["JOKER_red"]]
    elif game == "Macao":
        bad_labels = []

    boxes = detection["boxes"]
    bad_boxes = set()
    scores = detection["scores"]
    labels = detection["labels"]
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if label in bad_labels:
            bad_boxes.add(str(boxes[idx])) # use boxes coordinates because they are the only unique property of bounding boxes

    good_boxes = []
    good_scores = []
    good_labels = []
    for box, score, label in zip(boxes, scores, labels):
        if str(box) not in bad_boxes:
            good_boxes.append(box)
            good_scores.append(score)
            good_labels.append(label)
    if len(good_boxes) > 0:
        detection["boxes"] = torch.stack(good_boxes)
        detection["scores"] = torch.stack(good_scores)
        detection["labels"] = torch.stack(good_labels)
    else:
        detection["boxes"] = torch.as_tensor([])
        detection["scores"] = torch.as_tensor([])
        detection["labels"] = torch.as_tensor([])