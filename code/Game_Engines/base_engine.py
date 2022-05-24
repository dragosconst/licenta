from typing import List
import abc


class BaseEngine(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def update_detections(self, detected_player_hand: List, detected_card_pot: List):
        pass

    @abc.abstractmethod
    def act(self):
        pass
