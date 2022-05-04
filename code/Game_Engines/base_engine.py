from typing import List

class BaseEngine():
    def __init__(self):
        pass

    def update_detections(self, detected_player_hand: List, detected_card_pot: List):
        pass

    def act(self):
        pass