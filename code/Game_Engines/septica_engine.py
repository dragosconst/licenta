from typing import List
from enum import Enum, auto
from time import time
import random

from Game_Engines.base_engine import BaseEngine
from Models.RL.Envs.septica_utils import play_value
from Data_Processing.Raw_Train_Data.raw import pos_cls_inverse


class SepticaStates(Enum):
    THINK = auto()
    WAIT = auto()
    EVAL = auto()
    WORS = auto()


class SepticaEngine(BaseEngine):
    def __init__(self, septica_agent):
        super().__init__()
        self.agent = septica_agent

        self.state = SepticaStates.THINK
        self.player_hand = []
        self.cards_down = []
        self.played_card = None
        self.player_score = 0
        self.adv_score = 0

        # check if a valid change has just been made to the deck
        self.valid_change = False
        # for keeping track of splits
        self.splits_left = 0
        self.split_values = []

        self.is_first_player = True
        self.is_challenged = False
        self.used_cards = set()
        self.finished_time = 0
        self.WAIT_PERIOD = 6  # wait period between turns in seconds

        # cool statistics
        self.total_wins = 0
        self.total_matches = 0
        self.total_draws = 0
        self.total_losses = 0

    def update_detections(self, detected_player_hand: List, detected_card_down: str):
        """
        Function that handles updating the engine's detections.

        :param detected_player_hand: list of labels of detected player hand
        :param detected_card_pot: list of labels of detected card pot
        :return: nothing
        """
        if time() - self.finished_time < self.WAIT_PERIOD:
            return
        if len(detected_card_down) == 0 and self.state == SepticaStates.WORS and len(self.cards_down) > 0:
            # means adversary choose to end turn
            self.state = SepticaStates.EVAL
            return

        if len(detected_card_down) > 0:
            detected_card_down = [card for card in detected_card_down if card not in self.cards_down and card not in self.used_cards]
            if len(detected_card_down) > 0:
                detected_card_down = detected_card_down[0]
                if detected_card_down in self.used_cards:
                    print(f"Something went wrong with detections....")
                    print("-"*50)
                    print("-"*50)
            else:
                detected_card_down = None
        else:
            detected_card_down = None
        # for blackjack, we only need to check the hand each time a new card is drawn (either by us or the dealer)
        if (len(self.player_hand) != len(detected_player_hand) \
            or set(self.player_hand) != set(detected_player_hand) or (len(self.cards_down) > 0 and self.cards_down[-1] != detected_card_down)\
            or len(self.cards_down) == 0)\
                and self.state != SepticaStates.EVAL:
            if self.state != SepticaStates.WAIT and self.state != SepticaStates.WORS:
                self.player_hand = detected_player_hand
            if detected_card_down is not None and detected_card_down not in self.cards_down and detected_card_down != self.played_card:
                self.valid_change = True
            if detected_card_down is not None and (len(self.cards_down) == 0 or detected_card_down != self.cards_down[-1]):
                self.cards_down.append(detected_card_down)
                self.used_cards.update(self.cards_down)
                if self.state != SepticaStates.WORS and self.state != SepticaStates.WAIT:
                    print(f"Bad value for dealer detected. - reset recommended")
                    print("-"*75)

    def act(self):
        if time() - self.finished_time < self.WAIT_PERIOD:
            return
        if self.player_score + self.adv_score == 8:
            if self.player_score > self.adv_score:
                print(f"Won match with {self.player_score} points.")
            elif self.player_score == self.adv_score:
                print(f"Draw.")
            else:
                print(f"Lost to adversary. I have {self.player_score} points.")
            self.player_score = -1
            print(f"Reset agent for a new match.")
            return
        if self.player_score == -1:
            # game ended, nothing can happen anymore
            return

        if self.is_first_player:
            if self.state == SepticaStates.THINK:
                if len(self.player_hand) == 0 and len(self.cards_down) == 0:
                    return
                self.valid_change = False
                state = (self.player_hand, self.cards_down[0] if len(self.cards_down) > 0 else None, self.used_cards,\
                         play_value(self.cards_down), self.is_first_player, self.is_challenged)
                state = self.agent.process_state(state)
                action, extra_info = self.agent.get_action([state], eps=0)[0]
                if action == 0:
                    print(f"Put down card {extra_info}.")
                    print(f"My cards are {self.player_hand}.")
                    print(f"Cards down are {self.cards_down}.")
                    print(f"State is {state}")
                    if self.is_challenged and (extra_info[0] == "7" or extra_info[0] == self.cards_down[0][0]):
                        self.is_challenged = False
                    self.state = SepticaStates.WAIT
                    self.played_card = extra_info
                    self.cards_down.append(extra_info)
                    self.used_cards.add(self.played_card)
                elif action == 1:
                    print(f"End turn.")
                    print(f"My cards are {self.player_hand}.")
                    print(f"Cards down are {self.cards_down}.")
                    print(f"State is {state}")
                    self.state = SepticaStates.EVAL
            elif self.state == SepticaStates.WAIT:
                if self.valid_change:
                    print(f"My cards are {self.player_hand}.")
                    print(f"Cards down are {self.cards_down}.")
                    self.state = SepticaStates.THINK
                    if self.cards_down[-1][0] == "7" or self.cards_down[-1][0] == self.cards_down[0][0]:
                        self.is_challenged = True
                    self.used_cards.update(self.cards_down)
        else:
            if self.state == SepticaStates.WORS:
                stop = None
                # if len(self.cards_down) > 1:
                #     stop = input(f"Do you choose to end the game? y/n")
                # if stop is not None and stop[0].lower() == "y":
                #     self.state = SepticaStates.EVAL
                # else:
                if self.valid_change and len(self.cards_down) > 0:
                    print("waiting my dude")
                    print(f"My cards are {self.player_hand}.")
                    print(f"Cards down are {self.cards_down}.")
                    self.valid_change = False
                    if len(self.cards_down) == 1 or (self.cards_down[-1][0] == "7" or self.cards_down[-1][0] == self.cards_down[0][0]):
                        self.is_challenged = True
                    self.state = SepticaStates.THINK
                    self.used_cards.update(self.cards_down)
            elif self.state == SepticaStates.THINK:
                if len(self.player_hand) == 0:
                    return
                state = (self.player_hand, self.cards_down[0] if len(self.cards_down) > 0 else None, self.used_cards,\
                         play_value(self.cards_down), self.is_first_player, self.is_challenged)
                state = self.agent.process_state(state)
                action, extra_info = self.agent.get_action_no_eps([state], eps=0)[0]
                print(f"Put down card {extra_info}.")
                print(f"My cards are {self.player_hand}.")
                print(f"Cards down are {self.cards_down}.")
                print(f"State is {state}")
                if extra_info[0] == "7" or extra_info[0] == self.cards_down[0][0]:
                    self.is_challenged = False
                self.state = SepticaStates.WORS
                self.cards_down.append(extra_info)
                self.played_card = extra_info
                self.used_cards.add(self.played_card)
                self.finished_time = time() - self.WAIT_PERIOD//2
        if self.state == SepticaStates.EVAL:
            won = not self.is_challenged
            if won:
                print(f"Won {play_value(self.cards_down)} points.")
                self.player_score += play_value(self.cards_down)
            else:
                print(f"Lost {play_value(self.cards_down)} points.")
                self.adv_score += play_value(self.cards_down)
            self.is_first_player = won
            if self.is_first_player:
                self.state = SepticaStates.THINK
                self.is_challenged = False
            else:
                self.state = SepticaStates.WORS
                self.is_challenged = True
            self.cards_down = []
            self.played_card = None
            self.finished_time = time()
