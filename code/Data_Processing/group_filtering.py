from typing import Tuple, List, Dict, Union

import torch
import numpy as np


def get_closest_groups(groups: Dict[int, int], detections: Dict[str, torch.Tensor]) -> Tuple[List[int], ...]:
    closest_mean = None
    second_closest = None
    closest_gr_index = None
    second_closest_gr_index = None

    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    unique_vals = set()
    for key, val in groups.items():
        unique_vals.add(val)

    # check the mean y value of all groups detected
    for group in unique_vals:
        box_distances = []
        for key, val in groups.items():
            if val == group:
                box = boxes[key]
                box_distances.append(box[1]) # append y1
        box_distances = sum(box_distances)/len(box_distances)
        if closest_mean is None or box_distances > closest_mean: # get the group closest to the bottom of the capture
            if closest_mean is not None:
                second_closest = closest_mean
                second_closest_gr_index = closest_gr_index
            closest_mean = box_distances
            closest_gr_index = group
        elif second_closest is None or box_distances > second_closest:
            second_closest = box_distances
            second_closest_gr_index = group

    player_group = []
    cards_pot = []
    for key, val in groups.items():
        if val == closest_gr_index:
            player_group.append(key)
        elif val == second_closest_gr_index:
            cards_pot.append(key)
    return cards_pot, player_group

def new_indices(cards_pot: List[int], player_hand: List[int]) -> Tuple[List[int], List[int]]:
    """
    Get indices after filtering detections
    :param cards_pot: old cards pot indices
    :param player_hand: old player hand indices
    :return: tuple of new indices for both of them
    """
    smallest_key = 0
    all_indices = sorted(cards_pot + player_hand)
    keys_dict = {}
    for i in all_indices:
        keys_dict[i] = smallest_key
        smallest_key += 1

    for idx, card in enumerate(cards_pot):
        cards_pot[idx] = keys_dict[card]
    for idx, card in enumerate(player_hand):
        player_hand[idx] = keys_dict[card]
    return cards_pot, player_hand


def get_player_hand(game: str, detections: Dict[str, torch.Tensor]) -> Union[Tuple[List[int], ...], None]:
    """
    Get the groups consisting of the player hand and cards pot.
    :param game: what game we're playing
    :param detections: detections returned by the model
    :return: tuple of cards pot and player cards indices, the detections are filtered in-place
    """
    if game == "Solitaire": # solitaire works completely different
        return
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    def distance_between_boxes(b1, b2):
        x11, y11, x12, y12 = box1.cpu()
        x1c = (x11 + x12) / 2
        y1c = (y11 + y12) / 2
        x21, y21, x22, y22 = box2.cpu()
        x2c = (x21 + x22) / 2
        y2c = (y21 + y22) / 2

        return np.sqrt((x2c - x1c) ** 2 + (y2c - y1c) ** 2)

    RADIUS = 300
    gr_no = 0
    groups = {}
    for idx, (box1, label1, score1) in enumerate(zip(boxes, labels, scores)):
        min_dist = None
        if idx not in groups:
            groups[idx] = gr_no
            gr_no += 1
        for idy, (box2, label2, score2) in enumerate(zip(boxes, labels, scores)):
            if idx == idy:
                continue
            dist = distance_between_boxes(box1, box2)
            if dist <= RADIUS:
                if idy not in groups:
                    groups[idy] = groups[idx]
                elif groups[idx] != groups[idy]:
                    idy_group = groups[idy]
                    for k, v in groups.items(): # union between groups
                        if v == idy_group:
                            groups[k] = groups[idx]

    cards_pot, player_group = get_closest_groups(groups, detections)
    good_indices = sorted(cards_pot + player_group)
    detections["boxes"] = detections["boxes"][good_indices]
    detections["scores"] = detections["scores"][good_indices]
    detections["labels"] = detections["labels"][good_indices]

    return new_indices(cards_pot, player_group)
