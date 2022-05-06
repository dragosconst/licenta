from typing import List
from enum import Enum, auto
from time import time

from Models.RL.blackjack_simple import MCAgent
from Models.RL.Envs.blackjack_splitting import BlackjackEnvSplit, sum_hand, usable_ace
from Game_Engines.base_engine import BaseEngine


class BJStates(Enum):
    """
    State-machine like enums for transitions. Usual execution flow goes like:
    RESETTING->THINKING->H\ST\D\SP->T->H\ST\D\SP->...->W->D->Resetting
    it's possible to skip the W->D chaing by busting, in which case we have:
    R->T->...->R directly
    in the case of a blackjack, we'd have:
    R->T->R, since we instantly win -> also skip calling the agent in this case
    """
    THINKING = auto() # initial state it is at the start of the game or after hitting\splitting\doubling
    HITTING = auto()
    DOUBLING = auto()
    SPLITTING = auto()
    WAITING_FOR_DEALER = auto() # waiting for dealer to reach a score of 17
    DECIDING = auto() # state it switches to in the instant that the dealer passes (or reaches) 17
    RESETTING = auto() # state in-between games

class BlackjackEngine(BaseEngine):
    def __init__(self, bj_agent, statistics: bool=True):
        super().__init__()
        self.agent = bj_agent

        self.state = BJStates.RESETTING
        self.player_hand = []
        self.dealer_hand = []

        # check if a valid change has just been made to the deck
        self.valid_change = False
        # for keeping track of splits
        self.splits_left = 0
        self.split_values = []

        self.finished_time = 0
        self.WAIT_PERIOD = 8 # wait period between turns in seconds

        # cool statistics
        self.statistics = statistics
        self.total_wins = 0
        self.total_matches = 0
        self.total_draws = 0
        self.total_losses = 0

    def update_detections(self, detected_player_hand: List, detected_card_pot: List):
        """
        Function that handles updating the engine's detections.

        :param detected_player_hand: list of labels of detected player hand
        :param detected_card_pot: list of labels of detected card pot
        :return: nothing
        """
        if time() - self.finished_time < self.WAIT_PERIOD:
            return
        detected_player_hand = [1 if card[:-1] == "A" else card[:-1] for card in detected_player_hand]
        detected_card_pot = [1 if card[:-1] == "A" else card[:-1] for card in detected_card_pot]
        # for blackjack, we only need to check the hand each time a new card is drawn (either by us or the dealer)
        if len(self.player_hand) != len(detected_player_hand) or len(self.dealer_hand) != len(detected_card_pot)\
                or self.player_hand != detected_player_hand or self.dealer_hand != detected_card_pot: # for speed
            self.player_hand = detected_player_hand
            if detected_card_pot != self.dealer_hand:
                self.dealer_hand = detected_card_pot
                if self.state != BJStates.WAITING_FOR_DEALER and self.state != BJStates.RESETTING:
                    print(f"Bad value for dealer detected.")
                    print("-"*75)
                    self.state = BJStates.RESETTING
            self.valid_change = True

    def act(self):
        """
        Main function of the engine, here it decides what to do, based on the state it is in and the detections it has
        received.

        :return: nothing
        """

        # a game just ended or beginning first game, need to reset everything basically
        if self.state == BJStates.RESETTING:
            if time() - self.finished_time < self.WAIT_PERIOD:
                return
            if len(self.player_hand) == 0 or len(self.dealer_hand) == 0 or not self.valid_change:
                # if there's no (new) detection, don't do anything yet (maybe the dealer is flushing the deck etc.)
                return
            print(f"New hand.")
            if self.splits_left > 0:
                self.splits_left -= 1
            else:
                self.split_values = []
            self.state = BJStates.THINKING
            self.total_matches += 1
            self.valid_change = False # there will always be a valid change at the beginning of a hand
        # -----------------------------------
        elif self.state == BJStates.THINKING:
            # we end up in this state whenever we need to take a new decision
            splittable = None
            if len(self.player_hand) == 2 and self.player_hand[0] == self.player_hand[1]:
                splittable = 10 if self.player_hand[0] in {"K", "Q", "J"} else int(self.player_hand[0])
            sum_player = sum_hand(self.player_hand)
            if len(self.player_hand) == 2 and sum_player == 21: # natural blackjack
                if self.split_values == 0:
                    self.state = BJStates.DECIDING
                else:
                    print(f"Blackjack - from split. change hand")
                    self.split_values.append(sum_hand(self.player_hand))
                    self.finished_time = time()
                    self.state = BJStates.RESETTING
                return
            ace = usable_ace(self.player_hand)
            state = (splittable, sum_player, 10 if self.dealer_hand[0] in {"K", "Q", "J"}
                     else int(self.dealer_hand[0]), ace)

            action = self.agent.getAction(state)
            if action == 0:  # STAND
                print(f"I'm standing.")
                print(f"My cards are {self.player_hand}.")
                print(f"Dealer cards are {self.dealer_hand}.")
                self.state = BJStates.WAITING_FOR_DEALER
            elif action == 1:  # HIT
                print(f"Hit me.")
                print(f"My cards are {self.player_hand}.")
                print(f"Dealer cards are {self.dealer_hand}.")
                self.state = BJStates.HITTING
            elif action == 2:  # SPLIT
                print(f"Split.")
                print(f"My cards are {self.player_hand}.")
                print(f"Dealer cards are {self.dealer_hand}.")
                self.state = BJStates.SPLITTING
            elif action == 3:  # DOUBLE
                print(f"Double")
                print(f"My cards are {self.player_hand}.")
                print(f"Dealer cards are {self.dealer_hand}.")
                self.state = BJStates.DOUBLING
            if self.splits_left > 0 and self.state == BJStates.WAITING_FOR_DEALER:
                self.finished_time = time()
                self.state = BJStates.RESETTING
        # ---------------------------------------------
        elif self.state == BJStates.WAITING_FOR_DEALER:
            # just waiting for the dealer to reach a sum of 17
            dealer_sum = sum_hand(self.dealer_hand)
            if dealer_sum > 17 or (dealer_sum == 17 and not usable_ace(self.dealer_hand)):
                self.state = BJStates.DECIDING
                print(f"Dealer reached over 17.")
                print(f"My cards are {self.player_hand}.")
                print(f"Dealer cards are {self.dealer_hand}.")
        # ---------------------------------
        elif self.state == BJStates.HITTING:
            if self.valid_change:
                self.valid_change = False
                sum_player = sum_hand(self.player_hand)
                if sum_player > 21:
                    print(f"Busted.")
                    self.total_losses += 1
                    self.finished_time = time()
                    self.state = BJStates.RESETTING
                else:
                    self.state = BJStates.THINKING
        # ------------------------------------
        elif self.state == BJStates.SPLITTING:
            self.splits_left += 2 - (self.splits_left == 0) # if we already split, then a new split in fact adds only one new hand
            self.finished_time = time()
            self.state = BJStates.RESETTING
        # -----------------------------------
        elif self.state == BJStates.DOUBLING:
            if self.valid_change:
                self.valid_change = False
                sum_player = sum_hand(self.player_hand)
                if sum_player > 21:
                    print(f"Busted.")
                    self.total_losses += 1
                    self.finished_time = time()
                    self.state = BJStates.RESETTING
                else:
                    self.state = BJStates.WAITING_FOR_DEALER
                if self.splits_left > 0 and self.state == BJStates.WAITING_FOR_DEALER:
                    self.finished_time = time()
                    self.state = BJStates.RESETTING
        # -----------------------------------
        elif self.state == BJStates.DECIDING:
            if len(self.split_values) != 0:
                self.split_values.append(sum_hand(self.player_hand))

            sum_player = sum_hand(self.player_hand)
            sum_dealer = sum_hand(self.dealer_hand)
            if len(self.split_values) == 0:
                if sum_dealer > 21:
                    print(f"Dealer busted - no split.")
                    print(f"My cards are {self.player_hand}.")
                    print(f"Dealer cards are {self.dealer_hand}.")
                    self.total_wins += 1
                elif sum_player > sum_dealer:
                    print(f"I won.")
                    print(f"My cards are {self.player_hand}.")
                    print(f"Dealer cards are {self.dealer_hand}.")
                    self.total_wins += 1
                elif sum_player == sum_dealer:
                    print(f"Draw.")
                    print(f"My cards are {self.player_hand}.")
                    print(f"Dealer cards are {self.dealer_hand}.")
                    self.total_draws += 1
                elif sum_player < sum_dealer:
                    print(f"I lost.")
                    print(f"My cards are {self.player_hand}.")
                    print(f"Dealer cards are {self.dealer_hand}.")
                    self.total_losses += 1
            else:
                for sum_player in self.split_values:
                    if sum_player > 21:
                        continue
                    if sum_dealer > 21:
                        print(f"Dealer busted. - split")
                        self.total_wins += 1
                    elif sum_player > sum_dealer:
                        print(f"I won.")
                        print(f"My cards are {self.player_hand}.")
                        print(f"Dealer cards are {self.dealer_hand}.")
                        self.total_wins += 1
                    elif sum_player == sum_dealer:
                        print(f"Draw.")
                        print(f"My cards are {self.player_hand}.")
                        print(f"Dealer cards are {self.dealer_hand}.")
                        self.total_draws += 1
                    elif sum_player < sum_dealer:
                        print(f"I lost.")
                        print(f"My cards are {self.player_hand}.")
                        print(f"Dealer cards are {self.dealer_hand}.")
                        self.total_losses += 1

            print("-"*75)
            print("-"*75)
            self.state = BJStates.RESETTING
            self.finished_time = time()
