from typing import List, Optional, Dict, Tuple

from collections import defaultdict
import itertools
import torch

from Utils.utils import intersection_over_union

def eval(detection, target, iou_thresh: float=0.5):
    boxes = detection["boxes"]
    labels = detection["labels"]
    scores = detection["scores"]

    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for box, label, score in zip(boxes, labels, scores):
        max_box = None
        max_label = None
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            iou = intersection_over_union(box, gt_box)
            if iou >= iou_thresh:
                if max_box is None or iou > max_box:
                    max_box = gt_box
                    max_label = gt_label
        if max_box is None: # there shouldn't be anything here...
            false_positives += 1
            continue
        if max_label != label: # detected an objecy, but badly labeled it
            false_positives += 1
            print("bad label")
            continue
        true_positives += 1
    return true_positives, false_positives