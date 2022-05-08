from typing import List, Tuple
import random

from Models.RL.Envs.macao_utils import build_deck, draw_cards, draw_card, draw_hand,get_last_5_cards, get_card_suite, same_suite, shuffle_deck,\
                                        check_if_deck_empty

class MacaoRandom:
    def __init__(self, player_hand: List, adv_hand: List, cards_pot: List, deck: List, suite: List,
                 drawing_contest: int, turns_contest: int, player_turns: int, adv_turns: int, np_random, reward: int):
        self.player_hand = player_hand
        self.adversary_hand = adv_hand
        self.cards_pot = cards_pot
        self.deck = deck
        self.suite = suite
        self.player_turns = player_turns
        self.adv_turns = adv_turns
        self.drawing_contest = drawing_contest
        self.turns_contest = turns_contest
        self.np_random = np_random
        self.reward = reward
        self.full_suits = ["s", "c", "d", "h"]

    def check_legal_put(self, card):
        if self.drawing_contest > 0:
            return card[0] in {"2", "3", "4", "5"} or card[:3] == "jok"

        # now check if we are in a waiting turns contest, i.e. aces
        if self.turns_contest > 0:
            return card[0] == "A"

        # finally, we are left with the case of trying to put a regular card over another regular card
        return self.cards_pot[-1][0] == card[0] or same_suite(self.suite, card)

    def check_legal_pass(self):
        if len(self.deck) == 0 and len(self.cards_pot) == 1:
            return False
        return self.drawing_contest == 0 and self.turns_contest == 0

    def check_legal_concede(self):
        return self.drawing_contest > 0

    def check_legal_wait(self):
        return self.turns_contest > 0

    def check_legal_switch(self, card):
        return card[0] == "7" and self.turns_contest == 0 and self.drawing_contest == 0

    def has_to_draw(self):
        return self.drawing_contest > 0

    def has_to_wait(self):
        return self.turns_contest > 0

    def process_draws(self, card):
        if card[0] in {"2", "3"}:
            return int(card[0])
        if card[0] == "4":
            return -self.drawing_contest
        # 5 just passes it to adversary so don't do anything for it
        if card == "joker black":
            return 5
        if card == "joker red":
            return 10
        return 0

    def process_turns(self, card):
        if card[0] == "A":
            return 1
        return 0

    def get_legal_actions(self) -> List[Tuple]:
        """
        Get all legal actions from a given state.
        """
        new_hand = self.adversary_hand

        actions = []
        for action in [0, 1, 2, 3, 4, 5]:
            if action == 0:
                for card in new_hand:
                    if self.check_legal_put(card):
                        actions.append((action, card))
            elif action <= 3:
                if self.check_legal_pass() or self.check_legal_concede() or self.check_legal_wait():
                    actions.append((action, None))
            elif action == 4:
                for card in new_hand:
                    if card[0] == "7":
                        for suit in self.full_suits:
                            if self.check_legal_switch(card):
                                actions.append((action, card + " " + suit))
            elif action == 5:
                if self.adv_turns > 0 and self.drawing_contest == 0 and self.turns_contest == 0:
                    actions.append((action, None))
        return actions

    def get_action(self):
        actions = self.get_legal_actions()
        return random.sample(actions, 1)[0]

    def step(self):
        action, extra_info = self.get_action()

        if self.adv_turns > 0:
            self.adv_turns -= 1
        if action == 0:
            self.adversary_hand.remove(extra_info)
            self.cards_pot.append(extra_info)
            self.drawing_contest += self.process_turns(extra_info)
            self.reward -= 1
        elif action == 1:
            self.deck, self.cards_pot = check_if_deck_empty(self.deck, self.cards_pot)
            new_card, self.deck = draw_card(self.deck, self.np_random)
            self.adversary_hand.append(new_card)
            self.reward += 1
        elif action == 2:
            self.deck, self.cards_pot = check_if_deck_empty(self.deck, self.cards_pot)
            new_cards, self.deck = draw_cards(self.deck, self.cards_pot, self.drawing_contest, self.np_random)
            self.adversary_hand += new_cards
            self.reward += self.drawing_contest
            self.drawing_contest = 0
        elif action == 3:
            self.reward += self.turns_contest
            self.adv_turns += self.turns_contest
            self.turns_contest = 0
        elif action == 4:
            self.reward -= 1
            self.suite = extra_info[-1]
            self.adversary_hand.remove(extra_info[:2])
            self.cards_pot.append(extra_info[:2])
        elif action == 5:
            pass
