from typing import List, Tuple, Dict

import torch

def compare_detections(last_card_pot: List[int], last_player_hand: List[int], card_pot_idx: List[int],
                       player_hand_idx: List[int], detections: Dict[str, torch.Tensor]) \
                       -> Tuple[int, int, List[int], List[int]]:
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
    detected_card_pot = []
    identical_with_last_det_ph = True # flag to check if the last two hands are identical
    identical_with_last_det_cp = True # flag to check if the last two hands are identical

    boxes = detections["boxes"]
    labels = detections["labels"]
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        if idx in card_pot_idx:
            detected_card_pot.append(label.item())
            identical_with_last_det_cp = identical_with_last_det_cp and (label.item() in last_card_pot)
        elif idx in player_hand_idx:
            detected_player_hand.append(label.item())
            identical_with_last_det_ph = identical_with_last_det_ph and (label.item() in last_player_hand)

    flag_ph = 1 if identical_with_last_det_ph else -1
    flag_cp = 1 if identical_with_last_det_cp else -1
    return flag_cp, flag_ph, detected_card_pot, detected_player_hand