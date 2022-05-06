import random
from typing import List, Tuple

import gym
from gym import spaces
from gym.utils import seeding

from Models.RL.Envs.macao_minmax import alpha_beta, State, Game
from Models.RL.Envs.macao_utils import build_deck, draw_cards, draw_card, draw_hand,get_last_5_cards, get_card_suite, same_suite, shuffle_deck


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
        self.action_space = spaces.Discrete(5)
        self.seed()

        self.deck = build_deck()
        self.cards_pot = []
        self.suite = []

        self.player_hand = []
        self.adversary_hand = []
        self.player_turns_to_wait = 0
        self.adversary_turns_to_wait = 0
        self.card_just_put = False

    def reset(self):
        self.deck = build_deck()
        self.cards_pot = [draw_card(self.deck, self.np_random)]
        self.suite = get_card_suite(self.cards_pot[-1])

        self.player_hand = draw_hand(self.deck, self.np_random)
        self.adversary_hand = draw_hand(self.deck, self.np_random)
        self.player_turns_to_wait = 0
        self.adversary_turns_to_wait = 0
        self.card_just_put = True

    def _get_obs(self):
        return set(self.player_hand), get_last_5_cards(self.cards_pot), self.suite, self.card_just_put

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def check_legal_put(self, card):
        last_card_idx = len(self.cards_pot) - 1
        # check if there's a drawing contest going on
        while self.card_just_put and last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "5":
            # skip over redirects
            last_card_idx -= 1
        if self.card_just_put and last_card_idx >= 0 and (self.cards_pot[last_card_idx][0] in {"2", "3"} or self.cards_pot[last_card_idx][:3] \
            == "jok"):
            # check if we are doing a valid contestation
            return card[0] in {"2", "3", "4", "5"} or card[:3] == "jok"

        # now check if we are in a waiting turns contest, i.e. aces
        if self.card_just_put and self.cards_pot[last_card_idx][0] == "A":
            return card[0] == "A"

        # check for the special case of trying to put down a joker as a beginning of a contest
        if card[:3] == "jok":
            if card == "joker black":
                return self.cards_pot[last_card_idx][-1] in {"s", "c"}
            elif card == "joker red":
                return self.cards_pot[last_card_idx][-1] in {"h", "d"}

        # finally, we are left with the case of trying to put a regular card over another regular card
        return self.cards_pot[last_card_idx][0] == card[0] or same_suite(self.suite, card)

    def has_to_draw(self):
        if not self.card_just_put:
            return False
        return self.cards_pot[-1][0] in {"2", "3"} or self.cards_pot[-1][:3] == "jok"

    def has_to_wait(self):
        if not self.card_just_put:
            return False
        return self.cards_pot[-1][0] == "A"

    def build_state_from_env(self):
        game = Game(self.player_hand, self.adversary_hand, self.cards_pot, self.deck, self.suite, self.player_turns_to_wait,
                    self.adversary_turns_to_wait, self.card_just_put, self.np_random, 0)
        return State(game_state=game, current_player=Game.MAXP, depth=2) # run min-max for 2 moves at first

    def final_state(self):
        """
        Check if we reached a final state, either by winning or losing.
        """
        if not self.has_to_wait() and not self.has_to_draw() and len(self.player_hand) == 0:
            return 20
        if len(self.adversary_hand) == 0:
            for card in self.player_hand:
                if (card[0] in {"2", "3"} or card[:3] == "jok") and self.has_to_draw():
                    return 0
                if card[0] == "A" and self.has_to_wait():
                    return 0
            return -20
        return 0

    def step(self, action, extra_info=None):
        assert action is None or self.action_space.contains(action)
        if self.player_turns_to_wait > 0:
            action = -1 # don't do anything if i don't have to draw or wait extra turns
            if self.has_to_draw():
                action = 2
            elif self.has_to_wait():
                action = 3
            self.player_turns_to_wait -= 1
        reward = 0
        if action == 0:
            # put down card
            assert extra_info is not None
            card = extra_info
            assert self.check_legal_put(card)
            reward += 1
            self.player_hand.remove(card)
            self.cards_pot.append(card)
            self.card_just_put = True
        elif action == 1:
            # draw card
            if len(self.deck) == 0:
                # deck empty, therefore shuffle cards pot
                new_deck = self.cards_pot[:-1]
                new_deck = shuffle_deck(new_deck)
                self.deck = new_deck
                self.cards_pot = self.cards_pot[-1]
            reward -= 1
            new_card = draw_card(self.deck, self.np_random)
            self.card_just_put = False
            self.player_hand.append(new_card)
        elif action == 2:
            # concede to start drawing cards
            cards_to_draw = 0
            last_card_idx = len(self.cards_pot) - 1
            while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "5":
                # skip over redirects
                last_card_idx -= 1
            while last_card_idx >= 0 and (self.cards_pot[last_card_idx][0] in {"2", "3"} or self.cards_pot[last_card_idx][:3] \
            == "jok"):
                if self.cards_pot[last_card_idx][0] in {"2", "3"}:
                    cards_to_draw += int(self.cards_pot[last_card_idx][0])
                elif self.cards_pot[last_card_idx] == "joker black":
                    cards_to_draw += 5
                elif self.cards_pot[last_card_idx] == "joker red":
                    cards_to_draw += 10
                last_card_idx -= 1
            new_cards = draw_cards(deck=self.deck, cards_pot=self.cards_pot, num=cards_to_draw, np_random=self.np_random)
            self.player_hand += new_cards
            self.card_just_put = False
            reward -= cards_to_draw
        elif action == 3:
            # concede to waiting for turns
            turns_to_wait = self.player_turns_to_wait
            last_card_idx = len(self.cards_pot) - 1
            while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "A":
                turns_to_wait += 1
                last_card_idx -= 1
            self.player_turns_to_wait = turns_to_wait
            self.card_just_put = False
            reward -= turns_to_wait
        elif action == 4:
            # change suite with a 7
            assert extra_info is not None
            assert extra_info[0] == "7"
            self.suite = [extra_info[-1]]
            self.card_just_put = True
            for card in self.player_hand:
                if same_suite(self.suite, card):
                    reward += 1

        # change to new suite after player actions
        self.suite = get_card_suite(self.cards_pot[-1])

        # let's take the following case:
        # adversary puts down a 2h and player has 2c
        # now, for whatever reason, player decides to concede
        # therefore, the adversary won, so we can end the calculation now
        final = self.final_state()
        done = final != 0

        if not done:
            new_state = alpha_beta(state=self.build_state_from_env()).best_next_state  # type: State
            # update env after min-max choice
            game_state = new_state.game_state
            self.adversary_hand = game_state.adversary_hand
            self.cards_pot = game_state.cards_pot
            self.deck = game_state.deck
            self.suite = game_state.suite
            self.adversary_turns_to_wait = game_state.adv_turns
            self.card_just_put = game_state.just_put_card

        final = self.final_state()
        done = final != 0

        return self._get_obs(), reward + final, done

    def render(self, mode="human"):
        print(f"Your hand is {self.player_hand}.")
        print(f"Dealer hand is {self.adversary_hand}.")
        print(f"Card pot is {self.cards_pot}.")
        action = None
        extra_info = None
        if self.player_turns_to_wait == 0:
            action = input("State your action (with extra info if req):")
            action = action.split(" ")
            if len(action) >= 2:
                extra_info = " ".join(action[1:])
            action = int(action[0])
        _, reward, done = self.step(action, extra_info)
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
