import os
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.toy_text.blackjack import BlackjackEnv, draw_hand, draw_card, is_bust, score

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "K", "Q", "J"]

def sum_special_char(hand):
    sum = 0
    for card in hand:
        sum += 10 if type(card) == str else card
    return sum


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum_special_char(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum_special_char(hand) + 10
    return sum_special_char(hand)


def cmp(a, b):
    if a > 21:
        return -1
    return float(a > b) - float(a < b)


def splittable(hand): # Can we split?
    return hand[0] == hand[1] and len(hand) == 2


class BlackjackEnvSplit(BlackjackEnv):
    """
    Similar to the parent blackjack env, but you can now split! This adds an extra dimension to the observation space
    of size 2, which tells whether this hand can be split or not. Splitting will result in two hands at the same time, so we
    will have to account for the difference in the observation space for this too.
    Splitting is action 2, and can be used an indefinite amount of times.
    The space went from DeckxDealerxAce to Splitsx(DeckxDealerxAcexSplittablexSplitted), more mathematically:
    Sx(32x11x2x2x2), where S is a variable number.
    The space itself is viewed as a stack; when the player decides to stop hitting, the dealer plays and the
    hand is then flushed out of the observation space
    Final reward is cumulative, i.e. the reward for splitting action accounts for rewards over all hands resulted.
    """
    def __init__(self, natural=False, sab=False):
        super().__init__(natural, sab)
        self.action_space = spaces.Discrete(3)
        self.observation_space = []

        self.player = None # current playing hand
        self.dealer = None # current dealer hand
        self.splitted = None # whether this hand comes from a splitted hand or not
        self.player_index = None # which of the decks drawn is the player card


    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), splittable(self.player), self.splitted)

    def reset(self, seed: Optional[int] = None, return_info = False):
        super().reset()
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        self.player_index = 0
        self.splitted = False
        self.observation_space.append((self.player, self.splitted))
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}


    def step(self, action):
        # hit, therefore add a card to the top of the stack, which is self.player
        if action == 1:
            self.player.append(draw_card(self.np_random))
            done = False
            if is_bust(self.player):
                reward = -1
                # reached last hand
                if self.player == self.observation_space[-1][0]:
                    done = True
                else: # move to next hand
                    self.player_index += 1
                    self.player = self.observation_space[self.player_index][0]
                    self.splitted = self.observation_space[self.player_index][1]
            else:
                reward = 0
        elif action == 2:
            done = False
            if not splittable(self.player):
                # treat splitting when not possible as worse stoping
                if self.player == self.observation_space[-1][0]:
                    done = True
                else:  # move to next hand
                    self.player_index += 1
                    self.player = self.observation_space[self.player_index][0]
                    self.splitted = self.observation_space[self.player_index][1]
                reward = -1 # discourage from trying to split when there's no splitting permitted
            else:
                card = self.player[0]
                hand1 = [card, draw_card(self.np_random)]
                hand2 = [card, draw_card(self.np_random)]
                pindex = self.player_index
                # remove player hand and add two new hands
                self.observation_space = self.observation_space[:pindex] + [(hand1, True)] + [(hand2, True)] + self.observation_space[(pindex+1):]
                self.player = self.observation_space[self.player_index][0]
                self.splitted = self.observation_space[self.player_index][1]

                reward = 0.0 # splitting shouldn't really be encouraged or discouraged when possible
        elif action == 0: # stand
            done = False
            if self.player == self.observation_space[-1][0]:
                done = True
            else: # move to next hand
                self.player_index += 1
                self.player = self.observation_space[self.player_index][0]
                self.splitted = self.observation_space[self.player_index][1]
            reward = 0

        # dealer's turn
        if done:
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            rewards = [cmp(score(hand), score(self.dealer)) for hand, split in self.observation_space]
            reward = sum(rewards)
        return self._get_obs(), reward, done, {}

