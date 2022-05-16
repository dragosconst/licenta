from typing import List, Tuple, Dict
import copy

from Models.RL.Envs.septica_utils import draw_card, draw_hand, build_deck, play_value, draw_until, REWARD_MULT

class SepticaMinmax():
    MINP = 0
    MAXP = 1

    def __init__(self, player_hand: List, adversary_hand: List, played_cards: List, deck: List, is_challenging: bool,
                 first_player: int, player_score: int, adversary_score: int, np_random, reward: int):
        self.player_hand = player_hand
        self.adversary_hand = adversary_hand
        self.played_cards = played_cards
        self.deck = deck
        self.is_challenging = is_challenging
        self.first_player = first_player
        self.player_score = player_score
        self.adversary_score = adversary_score
        self.np_random = np_random
        self.reward = reward

    def check_legal_put(self, card, player):
        if not self.is_challenging:
            # when there's no challenge going on, any card goes
            return self.first_player != player or (self.first_player == player and len(self.played_cards) == 0)
        if not self.first_player == player:
            # when there's a challenge going on, every player but the first player can put down anything
            return len(self.played_cards) > 0
        return card[0] == "7" or card[0] == self.played_cards[0][0]

    def card_is_challenge(self, card, new_played_cards):
        return card[0] == "7" or card[0] == new_played_cards[0][0] or len(new_played_cards) == 1

    def check_legal_end(self, player):
        return player == self.first_player and len(self.played_cards) > 0

    def next_player(self):
        return (self.first_player + 1) % 2

    def moves(self, player: int):
        all_moves = []
        if player == self.MAXP:
            for card in self.adversary_hand:
                if self.check_legal_put(card, player):
                    new_hand = [card_ for card_ in self.adversary_hand if card_ != card]
                    new_played_cards = self.played_cards + [card]
                    all_moves.append(SepticaMinmax(copy.deepcopy(self.player_hand), copy.deepcopy(new_hand), new_played_cards,
                                                   copy.deepcopy(self.deck), self.card_is_challenge(card, new_played_cards), self.first_player,
                                                   self.player_score, self.adversary_score, copy.deepcopy(self.np_random), self.reward))
            if self.check_legal_end(player):
                # means first player is MAXP
                new_player_score = self.player_score
                new_adversary_score = self.adversary_score
                if not self.is_challenging:
                    new_adversary_score = self.adversary_score + play_value(self.played_cards) * REWARD_MULT
                    add_reward = -play_value(self.played_cards) * REWARD_MULT
                else:
                    new_player_score = self.player_score + play_value(self.played_cards) * REWARD_MULT
                    add_reward = play_value(self.played_cards) * REWARD_MULT
                new_player_hand = copy.deepcopy(self.player_hand)
                new_adversary_hand = copy.deepcopy(self.adversary_hand)
                new_deck = copy.deepcopy(self.deck)
                new_np_random = copy.deepcopy(self.np_random)
                if len(self.deck) > 0:
                    if len(new_adversary_hand) != len(new_player_hand):
                        print("aefewgfewg32WF32")
                    new_player_hand, new_adversary_hand, new_deck = draw_until(new_deck, new_player_hand, new_adversary_hand, 4, new_np_random)
                all_moves.append(SepticaMinmax(new_player_hand, new_adversary_hand, [], new_deck, False,
                                               self.next_player(), new_player_score, new_adversary_score, copy.deepcopy(new_np_random),
                                               self.reward + add_reward))
        else:
            for card in self.player_hand:
                if self.check_legal_put(card, player):
                    new_hand = [card_ for card_ in self.player_hand if card_ != card]
                    new_played_cards = self.played_cards + [card]
                    all_moves.append(SepticaMinmax(copy.deepcopy(new_hand), copy.deepcopy(self.adversary_hand), new_played_cards,
                                                   copy.deepcopy(self.deck), self.card_is_challenge(card, new_played_cards), self.first_player,
                                                   self.player_score, self.adversary_score, copy.deepcopy(self.np_random), self.reward))
            if self.check_legal_end(player):
                # means first player is MINP
                new_player_score = self.player_score
                new_adversary_score = self.adversary_score
                if not self.is_challenging:
                    new_player_score = self.player_score + play_value(self.played_cards) * REWARD_MULT
                    add_reward = play_value(self.played_cards) * REWARD_MULT
                else:
                    new_adversary_score = self.adversary_score + play_value(self.played_cards) * REWARD_MULT
                    add_reward = -play_value(self.played_cards) * REWARD_MULT
                new_player_hand = copy.deepcopy(self.player_hand)
                new_adversary_hand = copy.deepcopy(self.adversary_hand)
                new_deck = copy.deepcopy(self.deck)
                new_np_random = copy.deepcopy(self.np_random)
                if len(self.deck) > 0:
                    if len(new_adversary_hand) != len(new_player_hand):
                        print("saffwt13q")
                    new_player_hand, new_adversary_hand, new_deck = draw_until(new_deck, new_player_hand, new_adversary_hand, 4, new_np_random)
                all_moves.append(SepticaMinmax(new_player_hand, new_adversary_hand, [], new_deck, False,
                                               self.next_player(), new_player_score, new_adversary_score, copy.deepcopy(new_np_random),
                                               self.reward + add_reward))
        return all_moves

    def final_state(self):
        return len(self.deck) == 0 and len(self.player_hand) == 0 and len(self.adversary_hand) == 0 and len(self.played_cards) == 0

    def is_winning_hand(self, player):
        older_winning = self.is_challenging or len(self.played_cards) == 1
        return older_winning ^ (player != self.MAXP)

    def calculate_score(self, depth, player):
        return self.adversary_score - self.player_score

    @classmethod
    def adv_player(cls, player):
        return (player + 1) % 2


class SepticaState:
    def __init__(self, game_state: SepticaMinmax, current_player: int, depth: int, parent=None, score=None):
        self.game_state = game_state
        self.current_player = current_player

        self.depth = depth

        self.score = score

        self.possible_moves = []

        self.best_next_state = None

    def moves(self):
        l_moves = self.game_state.moves(self.current_player)
        adv_player = SepticaMinmax.adv_player(self.current_player)
        l_state_moves = [SepticaState(move, adv_player, self.depth - 1, parent=self) for move in l_moves]

        return l_state_moves


def alpha_beta(alpha: int=-500, beta: int=500, state: SepticaState=None):
    if state.depth == 0 or state.game_state.final_state():
        state.score = state.game_state.calculate_score(state.depth, player=state.current_player)
        return state

    if alpha > beta: # prunning
        return state

    state.possible_moves = state.moves()
    if len(state.possible_moves) == 0:
        print(state.game_state.player_hand)
        print(state.game_state.adversary_hand)
        print(state.game_state.played_cards)
        print(state.game_state.deck)
        print("????")

    if state.current_player == SepticaMinmax.MAXP:
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

    elif state.current_player == SepticaMinmax.MINP:
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

