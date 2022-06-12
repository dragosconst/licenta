from typing import List, Tuple, Dict

import torch


def compare_detections(last_card_pot: List[int], last_player_hand: List[int], card_pot_idx: List[int],
                       player_hand_idx: List[int], detections: Dict[str, torch.Tensor],
                       old_good_player: List[int], old_good_cards: List[int]) \
                       -> Tuple[int, int, List[int], List[int], List[int], List[int]]:
    """
    Check the current detections against the old ones. If they differ, the addition flag will be set to -1.
    :param last_card_pot: last card pot detected
    :param last_player_hand: last player hand detected
    :param card_pot_idx: card pot indexes in detections
    :param player_hand_idx: player hand indexes in detections
    :param detections: detections returned by model
    :return: tuple of two addition flags and two lists, representing the detected card pot and player hand.
    """
    detected_player_hand = []
    good_player_hand = []
    detected_card_pot = []
    good_cards_pot = []

    boxes = detections["boxes"]
    labels = detections["labels"]
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        if idx in card_pot_idx and label.item() in last_card_pot:
            good_cards_pot.append(label.item())
        if idx in card_pot_idx:
            detected_card_pot.append(label.item())
        elif idx in player_hand_idx and label.item() in last_player_hand:
            good_player_hand.append(label.item())
        if idx in player_hand_idx:
            detected_player_hand.append(label.item())

    flag_ph = 1 if len(old_good_player) == 0 or len(set(good_player_hand).intersection(set(old_good_player))) > 0 else -1
    flag_cp = 1 if len(old_good_cards) == 0 or len(set(good_cards_pot).intersection(set(old_good_cards))) > 0 else -1
    return flag_cp, flag_ph, good_cards_pot, good_player_hand, detected_card_pot, detected_player_hand
