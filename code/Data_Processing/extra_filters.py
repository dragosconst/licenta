from typing import List, Tuple, Dict

import torch

from Data_Processing.group_filtering import new_indices


def filter_same_card(detections: Dict[str, torch.Tensor], cards_pot: List[int], player_hand: List[int]) \
        -> Tuple[List[int], List[int]]:
    """
    In-place filtering of detections of the same card. Assumption is we only play games that require one standard deck.

    :param detections: detections returned by model
    :param cards_pot: cards pot indices
    :param player_hand: player hand indices
    :return: new indices for the card pot and player hand, in this order
    """

    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    labels_so_far_player = set()
    labels_so_far_card = set()
    good_indices = []
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if idx in cards_pot:
            labels_so_far = labels_so_far_player
        else:
            labels_so_far = labels_so_far_card
        if label.item() in labels_so_far:
            continue
        labels_so_far.add(label.item())
        good_indices.append(idx)

    new_pot = [idx for idx in cards_pot if idx in good_indices]
    new_hand = [idx for idx in player_hand if idx in good_indices]
    detections["boxes"] = boxes[good_indices]
    detections["labels"] = labels[good_indices]
    detections["scores"] = scores[good_indices]

    return new_indices(new_pot, new_hand)