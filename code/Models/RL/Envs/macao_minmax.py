from typing import List
import copy

from Models.RL.Envs.macao_utils import build_deck, draw_cards, draw_card, draw_hand,get_last_5_cards, get_card_suite, same_suite, shuffle_deck,\
                                        check_if_deck_empty

class Game:
    MINP = 0
    MAXP = 1

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

        self.full_deck = build_deck()

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
        return card[0] == 7 and self.turns_contest == 0 and self.drawing_contest == 0

    @classmethod
    def adv_player(cls, player):
        return cls.MAXP if player == cls.MINP else cls.MINP

    def has_to_draw(self):
        return self.drawing_contest > 0

    def has_to_wait(self):
        return self.turns_contest > 0

    def final_state(self):
        if len(self.player_hand) == 0 and not self.has_to_wait() and not self.has_to_draw():
            return self.MINP
        if len(self.adversary_hand) == 0 and not self.has_to_wait() and not self.has_to_draw():
            return self.MAXP
        return None

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
                                          get_card_suite(card), self.drawing_contest + self.process_draws(card),
                                          self.turns_contest + self.process_turns(card), self.player_turns, self.adv_turns, self.np_random, self.reward - 1))
                elif self.check_legal_switch(card) and self.adv_turns == 0:
                    new_adv_hand = [card_ for card_ in self.adversary_hand if card_ != card]
                    for suite in ["h", "c", "d", "s"]:
                        all_moves.append(Game(self.player_hand, new_adv_hand, copy.deepcopy(self.cards_pot + [card]), copy.deepcopy(self.deck),
                                          [suite], self.drawing_contest, self.turns_contest, self.player_turns, self.adv_turns, self.np_random, self.reward - 1))

            if self.check_legal_pass() and self.adv_turns == 0:
                new_deck = copy.deepcopy(self.deck)
                new_cards_pot = copy.deepcopy(self.cards_pot)
                new_deck, new_cards_pot = check_if_deck_empty(new_deck, new_cards_pot)
                new_card, new_deck = draw_card(new_deck, self.np_random)
                new_adv_hand = self.adversary_hand + [new_card]
                if self.adv_turns > 0:
                    self.adv_turns -= 1
                all_moves.append(Game(copy.deepcopy(self.player_hand), new_adv_hand, copy.deepcopy(new_cards_pot), copy.deepcopy(new_deck),
                                      copy.deepcopy(self.suite), self.drawing_contest, self.turns_contest, self.player_turns, self.adv_turns, self.np_random,
                                      self.reward + 1))
            if self.check_legal_concede():
                cards_to_draw = self.drawing_contest
                new_deck = copy.deepcopy(self.deck)
                new_cards_pot = copy.deepcopy(self.cards_pot)
                new_deck, new_cards_pot = check_if_deck_empty(new_deck, new_cards_pot)
                new_cards, new_deck = draw_cards(deck=new_deck, cards_pot=new_cards_pot, num=cards_to_draw, np_random=self.np_random)
                new_adv_hand = self.adversary_hand + new_cards
                if self.adv_turns > 0:
                    self.adv_turns -= 1
                all_moves.append(Game(copy.deepcopy(self.player_hand), new_adv_hand, copy.deepcopy(new_cards_pot),
                                      copy.deepcopy(new_deck), copy.deepcopy(self.suite), 0, self.turns_contest, self.player_turns,
                                      self.adv_turns, self.np_random, self.reward + cards_to_draw))
            elif self.check_legal_wait():
                turns_to_wait = self.adv_turns + self.turns_contest
                adv_turns_to_wait = turns_to_wait
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.drawing_contest, 0,
                                      self.player_turns, adv_turns_to_wait, self.np_random, self.reward + turns_to_wait))
            if self.adv_turns > 0 and not self.check_legal_wait() and not self.check_legal_concede():
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.drawing_contest, self.turns_contest,
                                      self.player_turns, self.adv_turns - 1, self.np_random, self.reward))
        elif player == self.MINP:
            for card in self.player_hand:
                if self.check_legal_put(card) and self.player_turns == 0:
                    new_player_hand = [card_ for card_ in self.player_hand if card_ != card]
                    all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot + [card]), copy.deepcopy(self.deck),
                                          get_card_suite(card), self.drawing_contest + self.process_draws(card), self.turns_contest,
                                          self.player_turns, self.adv_turns, self.np_random, self.reward + 1))
                elif self.check_legal_switch(card) and self.player_turns == 0:
                    new_player_hand = [card_ for card_ in self.player_hand if card_ != card]
                    for suite in ["h", "c", "d", "s"]:
                        all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot + [card]), copy.deepcopy(self.deck),
                                          [suite], self.drawing_contest, self.turns_contest, self.player_turns, self.adv_turns, self.np_random,
                                              self.reward + 1 + len([card for card in new_player_hand if same_suite([suite], card)])))

            if self.check_legal_pass() and self.player_turns == 0:
                new_deck = copy.deepcopy(self.deck)
                new_cards_pot = copy.deepcopy(self.cards_pot)
                new_deck, new_cards_pot = check_if_deck_empty(new_deck, new_cards_pot)
                new_card, new_deck = draw_card(new_deck, self.np_random)
                new_player_hand = self.player_hand + [new_card]
                all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(new_cards_pot), copy.deepcopy(new_deck),
                                      copy.deepcopy(self.suite), self.drawing_contest, self.turns_contest, self.player_turns,
                                      self.adv_turns, self.np_random, self.reward - 1))
            if self.check_legal_concede():
                cards_to_draw = self.drawing_contest
                new_deck = copy.deepcopy(self.deck)
                new_cards_pot = copy.deepcopy(self.cards_pot)
                new_deck, new_cards_pot = check_if_deck_empty(new_deck, new_cards_pot)
                new_cards, new_deck = draw_cards(deck=new_deck, cards_pot=new_cards_pot, num=cards_to_draw, np_random=self.np_random)
                new_player_hand = self.player_hand + new_cards
                if self.player_turns > 0:
                    self.player_turns -= 1
                all_moves.append(Game(new_player_hand, copy.deepcopy(self.adversary_hand), copy.deepcopy(new_cards_pot),
                                      copy.deepcopy(new_deck), copy.deepcopy(self.suite), 0, self.turns_contest, self.player_turns,
                                      self.adv_turns, self.np_random, self.reward - cards_to_draw))
            elif self.check_legal_wait():
                turns_to_wait = self.player_turns + self.turns_contest
                player_turns_to_wait = turns_to_wait
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.drawing_contest, 0,
                                      player_turns_to_wait, self.adv_turns, self.np_random, self.reward - turns_to_wait))
            if self.player_turns > 0 and not self.check_legal_wait() and not self.check_legal_concede():
                all_moves.append(Game(copy.deepcopy(self.player_hand), copy.deepcopy(self.adversary_hand), copy.deepcopy(self.cards_pot),
                                      copy.deepcopy(self.deck), copy.deepcopy(self.suite), self.drawing_contest, self.turns_contest,
                                      self.player_turns - 1, self.adv_turns, self.np_random, self.reward))

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
    if len(state.possible_moves) == 0:
        print("????")

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
