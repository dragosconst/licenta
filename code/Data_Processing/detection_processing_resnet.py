from typing import List, Optional, Dict, Tuple

import torch
import numpy as np

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
    """
    Apply nms over the new detections again. For reasons I'm not sure of, the net (frcnn specifically) seems to detect
    multiple times different cards in the same place, with the caveat that wrong detec  tions have (very) low probabilities
    compared to the good detection. By doing nms again we supress all such detections.

    :param detection: detections returned by net
    :return: nothing
    """

    boxes = detection["boxes"]
    bad_boxes = set()
    scores = detection["scores"]
    labels = detection["labels"]
    # the boxes are already sorted according to scores
    for idx, box in enumerate(boxes):
        for box2 in boxes[idx+1:]:
            if str(box2) in bad_boxes:
                continue
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
    """
    Filter all detections under a certain threshold. Just another way of removing possibly fake detections, real detections
    usually have >80% score, and very often >95%. Good detection with scores under 75% (the thresh I've chosen) are
    exceedingly rare.

    :param detection: detections returned by the net
    :return: nothing
    """
    boxes = detection["boxes"]
    bad_boxes = set()
    scores = detection["scores"]
    labels = detection["labels"]
    good_idx = []
    # the boxes are already sorted according to scores
    for idx, score in enumerate(scores):
        if score < 0.8:
            continue
        good_idx.append(idx)

    detection["boxes"] = detection["boxes"][good_idx]
    detection["labels"] = detection["labels"][good_idx]
    detection["scores"] = detection["scores"][good_idx]


def filter_detections_by_game(game: str, detection: Dict[str, torch.Tensor]) -> None:
    """
    In-place operation that removes impossible detections (i.e. Joker in a game that doesn't utilize Jokers) from
    detections. Useful for removing faulty joker detections and removing irrelevant card detections.

    :param game: string repesenting game name
    :param detection: detection given by model
    :return: nothing
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
    good_idx = []
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if label in bad_labels:
            continue
        good_idx.append(idx)

    detection["boxes"] = detection["boxes"][good_idx]
    detection["labels"] = detection["labels"][good_idx]
    detection["scores"] = detection["scores"][good_idx]


def filter_non_group_detections(game: str, detections: Dict[str, torch.Tensor]) -> None:
    """
    Filter detections that are too far away from any group of cards detected. "Groups" are defined as two or more
    cards close enough together. When checking distances in-between detections, we will ignore distances between detections
    of the same card, since a standard deck has only one instance of each card and this type of detections can occur only
    when detecting the card's symbol on another corner of the card, so it's not forming a group. Note this means that my
    model won't be able to play games that require multiple decks, but I believe there could be found solutions for them
    too, as a possible future extension of the program.
    If having a group with only one card is possible, only the one-card group with the highest score will be considered.

    :param game: the game being played, it's necessary for games like blackjack where a group can consist of only one card
    :param detections: detections returned by the model, they are supposed to be filtered beforehand using the other filters
    :return: nothing
    """
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    def distance_between_boxes(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Return distance from center of boxes to each other. Not perfect, but much easier to write than a comprehensive
        distance check.

        :param box1: box of (x1, y1, x2, y2) coords
        :param box2: box of (x1, y1, x2, y2) coords
        :return: distance float
        """

        x11, y11, x12, y12 = box1.cpu()
        x1c = (x11 + x12) / 2
        y1c = (y11 + y12) / 2
        x21, y21, x22, y22 = box2.cpu()
        x2c = (x21 + x22) / 2
        y2c = (y21 + y22) / 2

        return np.sqrt((x2c - x1c) ** 2 + (y2c - y1c) ** 2)

    if game == "Poker Texas Hold'Em": # in poker, there will never be a valid detection of only one card
        no_one_groups = True
    else: # all other games I've introduced have valid detections of only one card per group
        no_one_groups = False
    one_group = None
    good_indices = []
    MIN_DIST = 300 # minimum distance between x and one of the members of the group so that we may consider x a new member of the group
    for idx, (box1, label1, score1) in enumerate(zip(boxes, labels, scores)):
        min_dist = None
        for idy, (box2, label2, score2) in enumerate(zip(boxes, labels, scores)):
            if idx == idy:
                continue
            dist = distance_between_boxes(box1, box2)
            if dist > MIN_DIST:
                continue
            if min_dist is None or dist < min_dist:
                min_dist = dist
                break # since we are checking just for group membership, not actual group contents, we can break once we find a suitable card
        if min_dist is None and no_one_groups == False and one_group is None:
            one_group = idx
            good_indices.append(idx)
        if min_dist is not None:
            good_indices.append(idx)

    detections["boxes"] = detections["boxes"][good_indices]
    detections["labels"] = detections["labels"][good_indices]
    detections["scores"] = detections["scores"][good_indices]


def filter_small(detection: Dict[str, torch.Tensor], small_fact: float) -> None:
    boxes = detection["boxes"]
    scores = detection["scores"]
    labels = detection["labels"]

    good_idx = []
    too_small = small_fact * 10 ** 3
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if (x2 - x1) * (y2 - y1) < too_small:
            continue
        good_idx.append(idx)
    detection["boxes"] = detection["boxes"][good_idx]
    detection["labels"] = detection["labels"][good_idx]
    detection["scores"] = detection["scores"][good_idx]
