import os
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.envs.toy_text.blackjack import BlackjackEnv

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "K", "Q", "J"]

def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def sum_special_char(hand):
    sum = 0
    for card in hand:
        sum += 10 if card in {"K", "Q", "J"} else int(card)
    return sum


def usable_ace(hand):  # Does this hand have a usable ace?
    return ((1 in hand) or ('1' in hand)) and sum_special_char(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum_special_char(hand) + 10
    return sum_special_char(hand)

def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21

def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def cmp(a, b):
    return float(a > b) - float(a < b)


def splittable(hand): # Can we split?
    return len(hand) == 2 and hand[0] == hand[1]


class BlackjackEnvSplit(BlackjackEnv):
    """
    Similar to the parent blackjack env, but you can now split! This adds an extra dimension to the observation space
    of size 2, which tells whether this hand can be split or not. Splitting will result in two hands at the same time, so we
    will have to account for the difference in the observation space for this too.
    Splitting is action 2, and can be used an indefinite amount of times.
    Splitting results in spawning two new environments, and the reward of the split will be the cumulative reward given
    by the splits.
    The space will become of the form (15,32,11,2), where 15 represents the possible values for splitting: either None,
    in which case splitting is not permitted, or the value of the identical card.
    """
    def __init__(self, natural=False, sab=False):
        super().__init__(natural, sab)
        self.action_space = spaces.Discrete(3)

        self.player = None # current playing hand
        self.dealer = None # current dealer hand


    def get_split_val(self):
        if len(self.player) > 2:
            return None
        if self.player[0] != self.player[1]:
            return None
        return self.player[0]

    def _get_obs(self):
        return (self.get_split_val(), sum_hand(self.player), 10 if self.dealer[0] in {"K", "Q", "J"}
        else int(self.dealer[0]), usable_ace(self.player))

    def reset(self, seed: Optional[int] = None, return_info = False):
        super().reset()
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
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
                done = True
            else:
                reward = 0
        elif action == 2:
            done = False
            if not splittable(self.player):
                # treat splitting when not possible as worse stoping
                done = True
                reward = -100 # discourage from trying to split when there's no splitting permitted
                return self._get_obs(), reward, done, {}
            else:
                hand1 = BlackjackEnvSplit()
                hand1.reset()
                hand1.player[0] = self.player[0]
                hand2 = BlackjackEnvSplit()
                hand2.reset()
                hand2.player[0] = self.player[0]
                # this will return instead the two new hands for the agent to play
                return hand1, hand2
        elif action == 0: # stand
            done = True
            reward = 0

        # dealer's turn
        if done and reward >= 0:
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
        return self._get_obs(), reward, done, {}

