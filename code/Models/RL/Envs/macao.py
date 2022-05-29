import random
from typing import List, Tuple
import copy

import gym
from gym import spaces
from gym.utils import seeding

from Models.RL.Envs.macao_minmax import alpha_beta, State, MacaoMinmax
from Models.RL.Envs.macao_random import MacaoRandom
from Models.RL.Envs.macao_utils import build_deck, draw_cards, draw_card, draw_hand,get_last_5_cards, get_card_suite, same_suite, shuffle_deck,\
                                        check_if_deck_empty
# import Models.RL.macao_agent as ma


class MacaoEnv(gym.Env):
    """
    Macao env

    This env will simulate a 1-on-1 Macao game. The A.I. will use a version of MinMax to choose its next move.
    Macao rules:
    - both player start with 5 cards
    - if you can't put a card on the pot, then draw a card
    - you can put over the same suite or the same number\letter
    - A means you stay a turn
    - 4 stops you from having to draw
    - 5 passes the drawing to your adversary
    - 7 changes suite to whatever you want
    - 2 and 3 means draw 2 and 3 cards respectively
    - joker means draw 5 cards, red joker 10 cards
    - you are allowed to put any drawing card over any drawing card
    - when there aren't enough cards left to draw, reshuffle card pot, keeping only the last card down
    The action space is rather complex, an action will be described by either sending the descriptor of the card (i.e.
    2h, Ad etc.), saying "pass" when you can't put down any card, "concede" when you can't redirect drawings, "wait" if
    you concede to the A's put down and "7x y" for changing suite, where 7x is the 7's card you put down and y is the suite
    you wish to change to. Therefore, there are 5 different actions possible.
    Action table:
    index   | action name   |  extra_info              |
    ----------------------------------------------------
    0       | put card down | card suite and symbol    |
    1       | pass          | nothing                  |
    2       | concede       | nothing                  |
    3       | wait          | nothing                  |
    4       | change suite  | card suite and new suite |

    Reward system:
    passing: -1 reward
    putting down a card: +1 reward
    making opponent draw cards: +x reward, where x is amount of cards (on top of putting down cards reward)
    drawing cards bcs of opponent: -x reward, where x is amount of cards
    changing to favorable suite: +x reward, where x is amount of cards player holds of said suite (on top of putting down)
    making opponent wait turns: +x reward, where x is amount of turns
    waiting turns bcs of opponent: -x reward, where x is amount of turns
    winning: +20 reward
    losing: -20 reward
    - we also subtract the number of cards the dealer holds

    State:
    - set of player's cards
    - card pot card
    should try a test run without the bottom thing
    - known cards left in deck (reset when reshuffling deck)
    """

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(6)
        self.seed()

        self.deck = build_deck()
        self.cards_pot = []
        self.suite = []

        self.player_hand = []
        self.adversary_hand = []
        self.player_turns_to_wait = 0
        self.adversary_turns_to_wait = 0
        self.drawing_contest = 0
        self.turns_contest = 0

    def reset(self):
        self.deck = build_deck()
        self.cards_pot, self.deck = draw_card(self.deck, self.np_random)
        self.cards_pot = [self.cards_pot]
        self.suite = get_card_suite(self.cards_pot[-1])

        self.player_hand, self.deck = draw_hand(self.deck, self.np_random)
        self.adversary_hand, self.deck = draw_hand(self.deck, self.np_random)
        self.player_turns_to_wait = 0
        self.adversary_turns_to_wait = 0
        self.drawing_contest = 0
        if self.cards_pot[-1][0] in {"2", "3"}:
            self.drawing_contest = int(self.cards_pot[-1][0])
        elif self.cards_pot[-1] == "joker black":
            self.drawing_contest = 5
        elif self.cards_pot[-1] == "joker red":
            self.drawing_contest = 10
        self.turns_contest = 0
        if self.cards_pot[-1][0] == "A":
            self.turns_contest = 1

        return self._get_obs()

    def _get_obs(self):
        return set(self.player_hand), self.cards_pot[-1], self.cards_pot, self.drawing_contest, self.turns_contest, self.player_turns_to_wait,\
                self.adversary_turns_to_wait, len(self.adversary_hand), self.suite, \
               len(self.deck) == 0 and len(self.cards_pot) == 1


    def _get_adv_obs(self):
        return set(self.adversary_hand), self.cards_pot[-1], self.cards_pot, self.drawing_contest, self.turns_contest, self.adversary_turns_to_wait,\
                self.player_turns_to_wait, len(self.player_hand), self.suite, \
               len(self.deck) == 0 and len(self.cards_pot) == 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_legal_put(self, card):
        if self.drawing_contest > 0:
            return card[0] in {"2", "3", "4", "5"} or card[:3] == "jok"

        # now check if we are in a waiting turns contest, i.e. aces
        if self.turns_contest > 0:
            return card[0] == "A"

        # finally, we are left with the case of trying to put a regular card over another regular card
        return self.cards_pot[-1][0] == card[0] or same_suite(self.suite, card)

    def has_to_draw(self):
        return self.drawing_contest > 0

    def has_to_wait(self):
        return self.turns_contest > 0

    def build_state_from_env(self):
        game = MacaoMinmax(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot), \
                           copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.drawing_contest, self.turns_contest,
                           copy.deepcopy(self.player_turns_to_wait), copy.deepcopy(self.adversary_turns_to_wait),
                           copy.deepcopy(self.np_random), 0)
        return State(game_state=game, current_player=MacaoMinmax.MAXP, depth=2)  # run min-max for 2 moves at first

    def final_state(self):
        """
        Check if we reached a final state, either by winning or losing.
        """
        if not self.has_to_wait() and not self.has_to_draw() and len(self.player_hand) == 0:
            return 10 ** 2
        if len(self.adversary_hand) == 0 and not self.has_to_draw() and not self.has_to_wait():
            return -10 ** 2
        return 0

    def process_put_down(self, card):
        if card[0] in {"2", "3"}:
            self.drawing_contest += int(card[0])
        elif card[0] == "4":
            self.drawing_contest = 0
        # 5 just passes it to adversary so don't do anything for it
        elif card == "joker black":
            self.drawing_contest += 5
        elif card == "joker red":
            self.drawing_contest += 10
        elif card[0] == "A":
            self.turns_contest += 1

    def action_processing(self, action, extra_info=None):
        assert action is None or self.action_space.contains(action)
        reward = 0
        if action == 0:
            # put down card
            assert extra_info is not None
            card = extra_info
            assert self.check_legal_put(card)
            reward += 1
            self.player_hand.remove(card)
            self.cards_pot.append(card)
            self.process_put_down(card)
            # change to new suite after player actions
            self.suite = get_card_suite(self.cards_pot[-1])
        elif action == 1:
            # draw card
            self.deck, self.cards_pot = check_if_deck_empty(self.deck, self.cards_pot)
            reward -= 1
            new_card, self.deck = draw_card(self.deck, self.np_random)
            self.player_hand.append(new_card)
        elif action == 2:
            # concede to start drawing cards
            cards_to_draw = self.drawing_contest
            self.drawing_contest = 0
            self.deck, self.cards_pot = check_if_deck_empty(self.deck, self.cards_pot)
            new_cards, self.deck = draw_cards(deck=self.deck, cards_pot=self.cards_pot, num=cards_to_draw, np_random=self.np_random)
            self.player_hand += new_cards
            reward -= cards_to_draw
        elif action == 3:
            # concede to waiting for turns
            turns_to_wait = self.player_turns_to_wait + self.turns_contest
            self.turns_contest = 0
            self.player_turns_to_wait = turns_to_wait
            reward -= turns_to_wait
        elif action == 4:
            # change suite with a 7
            assert extra_info is not None
            assert extra_info[0] == "7"
            old_suits = self.suite
            self.suite = [extra_info[-1]]
            self.player_hand.remove(extra_info[:2])
            self.cards_pot.append(extra_info[:2])
            reward += 1
            if extra_info[-1] not in old_suits:  # avoid continously changing to the same suit
                for card in self.player_hand:
                    if card[0] != '7' and same_suite(self.suite, card): # reward for different cards
                        reward += 1
        elif action == 5:
            assert self.player_turns_to_wait > 0
        if self.player_turns_to_wait > 0:
            self.player_turns_to_wait -= 1
        return reward

    def step(self, action, extra_info=None):
        reward = self.action_processing(action, extra_info)

        # let's take the following case:
        # adversary puts down a 2h and player has 2c
        # now, for whatever reason, player decides to concede
        # therefore, the adversary won, so we can end the calculation now
        final = self.final_state()
        done = 1 if final != 0 else 0

        if not done:
            next_state = alpha_beta(state=self.build_state_from_env()).best_next_state  # type: State
            # update env after min-max choice
            game_state = next_state.game_state
            self.adversary_hand = game_state.adversary_hand
            self.cards_pot = game_state.cards_pot
            self.drawing_contest = game_state.drawing_contest
            self.turns_contest = game_state.turns_contest
            self.deck = game_state.deck
            self.suite = game_state.suite
            self.adversary_turns_to_wait = game_state.adv_turns
            reward += game_state.reward

        final = self.final_state()
        done = 1 if final != 0 else 0

        return self._get_obs(), reward + final, done

    def step_random(self, action, extra_info=None):
        reward = self.action_processing(action, extra_info)

        # let's take the following case:
        # adversary puts down a 2h and player has 2c
        # now, for whatever reason, player decides to concede
        # therefore, the adversary won, so we can end the calculation now
        final = self.final_state()
        done = 1 if final != 0 else 0

        if not done:
            mrandom = MacaoRandom(self.player_hand, self.adversary_hand, self.cards_pot, self.deck, self.suite,
                                  self.drawing_contest, self.turns_contest, self.player_turns_to_wait, self.adversary_turns_to_wait,
                                  self.np_random, 0)
            mrandom.step()
            self.adversary_hand = mrandom.adversary_hand
            self.cards_pot = mrandom.cards_pot
            self.deck = mrandom.deck
            self.suite = mrandom.suite
            self.drawing_contest = mrandom.drawing_contest
            self.turns_contest = mrandom.turns_contest
            self.adversary_turns_to_wait = mrandom.adv_turns
            reward += mrandom.reward

        final = self.final_state()
        done = 1 if final != 0 else 0

        return self._get_obs(), reward + final, done

    def step_agent(self, action, extra_info=None):
        reward = self.action_processing(action, extra_info)

        final = self.final_state()
        done = 1 if final != 0 else 0

        if not done:
            # agent = ma.get_macao_agent(self)
            agent = None
            action = agent.get_action([agent.process_state(self._get_adv_obs())], eps=0)[0]
            action, extra_info = action
            assert action is None or self.action_space.contains(action)
            reward = 0
            if action == 0:
                # put down card
                assert extra_info is not None
                card = extra_info
                assert self.check_legal_put(card)
                reward -= 1
                self.adversary_hand.remove(card)
                self.cards_pot.append(card)
                self.process_put_down(card)
                # change to new suite after player actions
                self.suite = get_card_suite(self.cards_pot[-1])
            elif action == 1:
                # draw card
                self.deck, self.cards_pot = check_if_deck_empty(self.deck, self.cards_pot)
                reward += 1
                new_card, self.deck = draw_card(self.deck, self.np_random)
                self.adversary_hand.append(new_card)
            elif action == 2:
                # concede to start drawing cards
                cards_to_draw = self.drawing_contest
                self.drawing_contest = 0
                self.deck, self.cards_pot = check_if_deck_empty(self.deck, self.cards_pot)
                new_cards, self.deck = draw_cards(deck=self.deck, cards_pot=self.cards_pot, num=cards_to_draw,
                                                  np_random=self.np_random)
                self.adversary_hand += new_cards
                reward += cards_to_draw
            elif action == 3:
                # concede to waiting for turns
                turns_to_wait = self.adversary_turns_to_wait + self.turns_contest
                self.turns_contest = 0
                self.adversary_turns_to_wait = turns_to_wait
                reward += turns_to_wait
            elif action == 4:
                # change suite with a 7
                assert extra_info is not None
                assert extra_info[0] == "7"
                self.suite = [extra_info[-1]]
                self.adversary_hand.remove(extra_info[:2])
                self.cards_pot.append(extra_info[:2])
                reward -= 1
                for card in self.player_hand:
                    if same_suite(self.suite, card):
                        reward -= 1
            elif action == 5:
                assert self.adversary_turns_to_wait > 0

            if self.adversary_turns_to_wait > 0:
                self.adversary_turns_to_wait -= 1

        final = self.final_state()
        done = 1 if final != 0 else 0

        return self._get_obs(), reward + final, done

    def render(self, mode="human"):
        print(f"Your hand is {self.player_hand}.")
        print(f"Dealer has {len(self.adversary_hand)} more cards.")
        # print(f"Dealer has {self.adversary_hand} more cards.")
        print(f"Suits is {self.suite}")
        print(f"Turns left is {self.player_turns_to_wait}")
        # print(f"Dealer hand is {self.adversary_hand}.")
        print(f"Card pot is {self.cards_pot}.")
        action = None
        extra_info = None
        if self.player_turns_to_wait == 0 or self.has_to_wait() or self.has_to_draw():
            action = input("State your action (with extra info if req):")
            action = action.split(" ")
            if len(action) >= 2:
                extra_info = " ".join(action[1:])
            action = int(action[0])
        _, reward, done = self.step_agent(action, extra_info)
        print(f"Pot after adv move: {self.cards_pot}.")
        print(f"You got a reward of {reward}.")
        print("-"*100)
        if done:
            print(f"Game done.")
        return done


if __name__ == "__main__":
    env = MacaoEnv()
    env.reset()

    while not env.render():
        continue

