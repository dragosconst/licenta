from typing import List
from enum import Enum, auto

from Models.RL.blackjack_simple import MCAgent
from Models.RL.Envs.blackjack_splitting import BlackjackEnvSplit, sum_hand, usable_ace


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

class BlackjackEngine:
    def __init__(self, bj_agent: MCAgent, statistics: bool=True):
        self.agent = bj_agent

        self.state = BJStates.RESETTING
        self.player_hand = []
        self.dealer_hand = []

        # check if a valid change has just been made to the deck
        self.valid_change = False
        # for keeping track of splits
        self.splits_left = 0
        self.split_values = []

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
        # for blackjack, we only need to check the hand each time a new card is drawn (either by us or the dealer)
        if len(self.player_hand) != len(detected_player_hand) or len(self.dealer_hand) != len(detected_card_pot): # for speed
            self.player_hand = [1 if card[:-1] == "A" else card[:-1] for card in detected_player_hand]
            self.dealer_hand = [1 if card[:-1] == "A" else card[:-1] for card in detected_card_pot]
            self.valid_change = True

    def act(self):
        """
        Main function of the engine, here it decides what to do, based on the state it is in and the detections it has
        received.

        :return: nothing
        """

        # a game just ended or beginning first game, need to reset everything basically
        if self.state == BJStates.RESETTING:
            if len(self.player_hand) == 0 or len(self.dealer_hand) == 0 or not self.valid_change:
                # if there's no (new) detection, don't do anything yet (maybe the dealer is flushing the deck etc.)
                return

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
                    self.split_values.append(sum_hand(self.player_hand))
                    self.state = BJStates.RESETTING
                return
            ace = usable_ace(self.player_hand)
            state = (splittable, sum_player, 10 if self.dealer_hand[0] in {"K", "Q", "J"}
                     else int(self.dealer_hand[0]), ace)

            action = self.agent.get_action(state)
            if action == 0:  # STAND
                self.state = BJStates.WAITING_FOR_DEALER
            elif action == 1:  # HIT
                self.state = BJStates.HITTING
            elif action == 2:  # SPLIT
                self.state = BJStates.SPLITTING
            elif action == 3:  # DOUBLE
                self.state = BJStates.DOUBLING
            if self.splits_left > 0 and self.state == BJStates.WAITING_FOR_DEALER:
                self.state = BJStates.RESETTING
        # ---------------------------------------------
        elif self.state == BJStates.WAITING_FOR_DEALER:
            # just waiting for the dealer to reach a sum of 17
            if self.splits_left > 0:
                self.split_values.append(sum_hand(self.player_hand))

            dealer_sum = sum_hand(self.dealer_hand)
            if dealer_sum >= 17:
                self.state = BJStates.DECIDING
        # ---------------------------------
        elif self.state == BJStates.HITTING:
            if self.valid_change:
                self.valid_change = False
                sum_player = sum_hand(self.player_hand)
                if sum_player > 21:
                    self.total_losses += 1
                    self.state = BJStates.RESETTING
                else:
                    self.state = BJStates.THINKING
        # ------------------------------------
        elif self.state == BJStates.SPLITTING:
            self.splits_left += 2 - (self.splits_left == 0) # if we already split, than a new split in fact adds only one new hand
            self.state = BJStates.RESETTING
        # -----------------------------------
        elif self.state == BJStates.DOUBLING:
            if self.valid_change:
                self.valid_change = False
                sum_player = sum_hand(self.player_hand)
                if sum_player > 21:
                    self.total_losses += 1
                    self.state = BJStates.RESETTING
                else:
                    self.state = BJStates.WAITING_FOR_DEALER
                if self.splits_left > 0 and self.state == BJStates.WAITING_FOR_DEALER:
                    self.state = BJStates.RESETTING
        # -----------------------------------
        elif self.state == BJStates.DECIDING:
            if len(self.split_values) != 0:
                self.split_values.append(sum_hand(self.player_hand))

            sum_player = sum_hand(self.player_hand)
            sum_dealer = sum_hand(self.dealer_hand)
            if len(self.split_values) == 0:
                if sum_dealer > 21:
                    self.total_wins += 1
                elif sum_player > sum_dealer:
                    self.total_wins += 1
                elif sum_player == sum_dealer:
                    self.total_draws += 1
                elif sum_player < sum_dealer:
                    self.total_losses += 1
            else:
                for sum_player in self.split_values:
                    if sum_dealer > 21:
                        self.total_wins += 1
                    elif sum_player > sum_dealer:
                        self.total_wins += 1
                    elif sum_player == sum_dealer:
                        self.total_draws += 1
                    elif sum_player < sum_dealer:
                        self.total_losses += 1

            self.state = BJStates.RESETTING
