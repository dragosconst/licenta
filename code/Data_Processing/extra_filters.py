from typing import List, Tuple, Dict

import torch


def filter_same_card(detections: Dict[str, torch.Tensor], cards_pot: List[int], player_hand: List[int]) \
        -> Tuple[List[int], List[int]]:
    """
    In-place filtering of detections of the same card. Assumption is we only play games that require one standard deck.
    :param detections: detections returned by model
    :return: nothing, it is an in-place operation
    """

    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    labels_so_far = set()
    good_indices = []
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if label.item() in labels_so_far:
            continue
        labels_so_far.add(label.item())
        good_indices.append(idx)

    new_pot = [idx for idx in cards_pot if idx in good_indices]
    new_hand = [idx for idx in player_hand if idx in good_indices]
    detections["boxes"] = boxes[good_indices]
    detections["labels"] = labels[good_indices]
    detections["scores"] = scores[good_indices]

    return new_pot, new_hand