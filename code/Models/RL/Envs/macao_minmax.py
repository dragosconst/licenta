from typing import List
import copy

from Models.RL.Envs.macao_utils import build_deck, draw_cards, draw_card, draw_hand,get_last_5_cards, get_card_suite, same_suite, shuffle_deck

class Game:
    MINP = 0
    MAXP = 1

    def __init__(self, player_hand: List, adv_hand: List, cards_pot: List, deck: List, suite: List,
                 player_turns: int, adv_turns: int, last_action: bool, np_random, reward: int):
        self.player_hand = player_hand
        self.adversary_hand = adv_hand
        self.cards_pot = cards_pot
        self.deck = deck
        self.suite = suite
        self.player_turns = player_turns
        self.adv_turns = adv_turns
        self.just_put_card = last_action
        self.np_random = np_random
        self.reward = reward

        self.full_deck = build_deck()

    def check_legal_put(self, card):
        last_card_idx = len(self.cards_pot) - 1
        # check if there's a drawing contest going on
        while self.just_put_card and last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "5":
            # skip over redirects
            last_card_idx -= 1
        if self.just_put_card and last_card_idx >= 0 and (self.cards_pot[last_card_idx][0] in {"2", "3"} or self.cards_pot[last_card_idx][:3] \
            == "jok"):
            # check if we are doing a valid contestation
            return card[0] in {"2", "3", "4", "5"} or card[:3] == "jok"

        # now check if we are in a waiting turns contest, i.e. aces
        if self.just_put_card and self.cards_pot[last_card_idx][0] == "A":
            return card[0] == "A"

        # check for the special case of trying to put down a joker as a beginning of a contest
        if card[:3] == "jok":
            if card == "joker black":
                return self.cards_pot[last_card_idx][-1] in {"s", "c"}
            elif card == "joker red":
                return self.cards_pot[last_card_idx][-1] in {"h", "d"}

        # finally, we are left with the case of trying to put a regular card over another regular card
        return self.cards_pot[last_card_idx][0] == card[0] or same_suite(self.suite, card)

    def check_legal_pass(self):
        last_card_idx = len(self.cards_pot) - 1
        # check if there's a drawing contest going on
        while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "5":
            # skip over redirects
            last_card_idx -= 1
        if last_card_idx >= 0 and (self.cards_pot[last_card_idx][0] in {"2", "3"} or self.cards_pot[last_card_idx][:3] \
            == "jok"):
            # check if we are doing a valid contestation
            return False
        if self.cards_pot[last_card_idx][0] == "A":
            return False
        return True

    def check_legal_concede(self):
        if not self.just_put_card:
            return False
        last_card_idx = len(self.cards_pot) - 1
        # check if there's a drawing contest going on
        while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "5":
            # skip over redirects
            last_card_idx -= 1
        return last_card_idx >= 0 and (self.cards_pot[last_card_idx][0] in {"2", "3"} or self.cards_pot[last_card_idx][:3] \
            == "jok")

    def check_legal_wait(self):
        if not self.just_put_card:
            return False
        return self.cards_pot[-1][0] == "A"

    def check_legal_switch(self, card):
        return card[0] == 7

    @classmethod
    def adv_player(cls, player):
        return cls.MAXP if player == cls.MINP else cls.MINP

    def final_state(self):
        if len(self.player_hand) == 0:
            return self.MINP
        if len(self.adversary_hand) == 0:
            return self.MAXP
        return None

    def moves(self, player):
        """
        Calculate all possible moves from the given deck.

        """
        all_moves = []
        if player == self.MAXP:
            # adversary's turn to play
            for card in self.adversary_hand:
                if self.check_legal_put(card) and self.adv_turns == 0:
                    new_adv_hand = [card_ for card_ in self.adversary_hand if card_ != card]
                    all_moves.append(Game(self.player_hand, new_adv_hand, copy.deepcopy(self.cards_pot + [card]), copy.deepcopy(self.deck),
                                          get_card_suite(card), self.player_turns, self.adv_turns, True, self.np_random, self.reward - 1))
                elif self.check_legal_switch(card) and self.adv_turns == 0:
                    new_adv_hand = [card_ for card_ in self.adversary_hand if card_ != card]
                    for suite in ["h", "c", "d", "s"]:
                        all_moves.append(Game(self.player_hand, new_adv_hand, copy.deepcopy(self.cards_pot + [card]), copy.deepcopy(self.deck),
                                          [suite], self.player_turns, self.adv_turns, True, self.np_random, self.reward - 1))

            if self.check_legal_pass() and self.adv_turns == 0:
                if len(self.deck) == 0:
                    new_deck = self.cards_pot[:-1]
                    new_deck = shuffle_deck(new_deck)
                    self.deck = new_deck
                    self.cards_pot = self.cards_pot[-1]
                new_card = draw_card(self.deck, self.np_random)
                new_adv_hand = self.adversary_hand + [new_card]
                if self.adv_turns > 0:
                    self.adv_turns -= 1
                all_moves.append(Game(copy.deepcopy(self.player_hand), new_adv_hand, copy.deepcopy(self.cards_pot), copy.deepcopy(self.deck),
                                      copy.deepcopy(self.suite), self.player_turns, self.adv_turns, False, self.np_random,
                                      self.reward + 1))
            if self.check_legal_concede():
                cards_to_draw = 0
                last_card_idx = len(self.cards_pot) - 1
                while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "5":
                    # skip over redirects
                    last_card_idx -= 1
                while last_card_idx >= 0 and (
                        self.cards_pot[last_card_idx][0] in {"2", "3"} or self.cards_pot[last_card_idx][:3]
                        == "jok"):
                    if self.cards_pot[last_card_idx][0] in {"2", "3"}:
                        cards_to_draw += int(self.cards_pot[last_card_idx][0])
                    elif self.cards_pot[last_card_idx] == "joker black":
                        cards_to_draw += 5
                    elif self.cards_pot[last_card_idx] == "joker red":
                        cards_to_draw += 10
                    last_card_idx -= 1
                new_cards = draw_cards(deck=self.deck, cards_pot=self.cards_pot, num=cards_to_draw, np_random=self.np_random)
                new_adv_hand = self.adversary_hand + new_cards
                if self.adv_turns > 0:
                    self.adv_turns -= 1
                all_moves.append(Game(copy.deepcopy(self.player_hand), new_adv_hand, copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.player_turns,
                                      self.adv_turns, False, self.np_random, self.reward + cards_to_draw))
            elif self.check_legal_wait():
                turns_to_wait = self.adv_turns
                last_card_idx = len(self.cards_pot) - 1
                while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "A":
                    turns_to_wait += 1
                    last_card_idx -= 1
                adv_turns_to_wait = turns_to_wait
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.player_turns, adv_turns_to_wait,
                                      False, self.np_random, self.reward + turns_to_wait))
            if self.adv_turns > 0 and not self.check_legal_wait() and not self.check_legal_concede():
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.player_turns, self.adv_turns - 1,
                                      False, self.np_random, self.reward))
        elif player == self.MINP:
            for card in self.player_hand:
                if self.check_legal_put(card) and self.player_turns == 0:
                    new_player_hand = [card_ for card_ in self.player_hand if card_ != card]
                    all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot + [card]), copy.deepcopy(self.deck),
                                          get_card_suite(card), self.player_turns, self.adv_turns, True, self.np_random,
                                          self.reward + 1))
                elif self.check_legal_switch(card) and self.player_turns == 0:
                    new_player_hand = [card_ for card_ in self.player_hand if card_ != card]
                    for suite in ["h", "c", "d", "s"]:
                        all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot + [card]), copy.deepcopy(self.deck),
                                          [suite], self.player_turns, self.adv_turns, True, self.np_random,
                                              self.reward + 1 + len([card for card in new_player_hand if same_suite([suite], card)])))

            if self.check_legal_pass() and self.player_turns == 0:
                if len(self.deck) == 0:
                    new_deck = self.cards_pot[:-1]
                    new_deck = shuffle_deck(new_deck)
                    self.deck = new_deck
                    self.cards_pot = self.cards_pot[-1]
                new_card = draw_card(self.deck, self.np_random)
                new_player_hand = self.player_hand + [new_card]
                if self.player_turns > 0:
                    self.player_turns -= 1
                all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot), copy.deepcopy(self.deck),
                                      copy.deepcopy(self.suite), self.player_turns, self.adv_turns, False, self.np_random,
                                      self.reward - 1))
            if self.check_legal_concede():
                cards_to_draw = 0
                last_card_idx = len(self.cards_pot) - 1
                while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "5":
                    # skip over redirects
                    last_card_idx -= 1
                while last_card_idx >= 0 and (
                        self.cards_pot[last_card_idx][0] in {"2", "3"} or self.cards_pot[last_card_idx][:3]
                        == "jok"):
                    if self.cards_pot[last_card_idx][0] in {"2", "3"}:
                        cards_to_draw += int(self.cards_pot[last_card_idx][0])
                    elif self.cards_pot[last_card_idx] == "joker black":
                        cards_to_draw += 5
                    elif self.cards_pot[last_card_idx] == "joker red":
                        cards_to_draw += 10
                    last_card_idx -= 1
                new_cards = draw_cards(deck=self.deck, cards_pot=self.cards_pot, num=cards_to_draw, np_random=self.np_random)
                new_player_hand = self.player_hand + new_cards
                if self.player_turns > 0:
                    self.player_turns -= 1
                all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.player_turns,
                                      self.adv_turns, False, self.np_random, self.reward - cards_to_draw))
            elif self.check_legal_wait():
                turns_to_wait = self.player_turns
                last_card_idx = len(self.cards_pot) - 1
                while last_card_idx >= 0 and self.cards_pot[last_card_idx][0] == "A":
                    turns_to_wait += 1
                    last_card_idx -= 1
                player_turns_to_wait = turns_to_wait
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), player_turns_to_wait, self.adv_turns,
                                      False, self.np_random, self.reward - turns_to_wait))
            if self.player_turns > 0 and not self.check_legal_wait() and not self.check_legal_concede():
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.player_turns - 1, self.adv_turns,
                                      False, self.np_random, self.reward))

        return all_moves

    # score is calculated to encourage putting down cards and winning contests against the player
    def calculate_score(self, depth):
        t_final = self.final_state()

        if t_final == self.__class__.MAXP:
            return depth + 10 ** 3
        elif t_final == self.__class__.MINP:
            return - depth - (10 ** 3)
        else:
            return len(self.player_hand)-len(self.adversary_hand)


class State:
    def __init__(self, game_state: Game, current_player: int, depth: int, parent=None, score=None):
        self.game_state = game_state
        self.current_player = current_player

        self.depth = depth

        self.score = score

        self.possible_moves = []

        self.best_next_state = None

    def moves(self):
        l_moves = self.game_state.moves(self.current_player)
        adv_player = Game.adv_player(self.current_player)
        l_state_moves = [State(move, adv_player, self.depth - 1, parent=self) for move in l_moves]

        return l_state_moves

    def __str__(self):
        sir = str(self.game_state) + "(Juc curent:" + str(self.current_player) + ")\n"
        return sir

    def __repr__(self):
        sir = str(self.game_state) + "(Juc curent:" + str(self.current_player) + ")\n"
        return sir


def alpha_beta(alpha: int=-500, beta: int=500, state: State=None):
    if state.depth == 0 or state.game_state.final_state():
        state.score = state.game_state.calculate_score(state.depth)
        return state

    if alpha > beta: # prunning
        return state

    state.possible_moves = state.moves()

    if state.current_player == Game.MAXP:
        current_score = float('-inf')

        for move in state.possible_moves:
            # calculeaza scorul
            new_state = alpha_beta(alpha, beta, move)

            if current_score < new_state.score:
                state.best_next_state = new_state
                current_score = new_state.score
            if alpha < new_state.score:
                alpha = new_state.score
                if alpha >= beta:
                    break

    elif state.current_player == Game.MINP:
        current_score = float('inf')

        for move in state.possible_moves:

            new_state = alpha_beta(alpha, beta, move)

            if current_score > new_state.score:
                state.best_next_state = new_state
                current_score = new_state.score

            if beta > new_state.score:
                beta = new_state.score
                if alpha >= beta:
                    break
    state.score = state.best_next_state.score
    return state
