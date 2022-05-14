import copy

import gym
from gym import spaces
from gym.utils import seeding

from Models.RL.Envs.septica_utils import draw_card, draw_hand, build_deck, play_value, draw_until
from Models.RL.Envs.septica_minmax import SepticaMinmax, alpha_beta, SepticaState


def deepc(object):
    return copy.deepcopy(object)


class SepticaEnv(gym.Env):
    """
    An env for the cards game Septica. It specifically simulates a 1v1 game.
    The goal of the game is to amass as many points as possible, by collecting the 10's and aces cards. Collecting cards
    is done by "taieturi", i.e. challenging the opposing player.
    The rules are the following:
    - the player must have 4 cards at all times until no longer possible (i.e. deck has run out)
    - the current player has to put a card down (any card is valid)
    - the next player can put ANY card down
    - if the next player puts down any card besides the same character (not suite!) as the FIRST card played during the
      current play or any 7 card, they concede the current play to the opposing player
    - if the next player puts down a card of any of the two categories aforementioned, they perform a "taietura", meaning
      that they challenge the current play
    - if a "taietura" is taking place, the first player can only further contest or has to otherwise concede the play, the
      other player(s) are allowed to put down any card untill the first player ends the play
    - if the next player is the player that initiated the play, they can additionally choose to end the play, in which
      case the cards go to whoever won the most recent challenge (i.e. last "taietura" played)
    - no player besides the first player can end the play
    - you cannot stack "taieturi", i.e. keep putting contesting cards over if no one further contested you during the
      current play
    - the only cards that award points are 10s and aces
    - the game is played with cards starting from the 7s (half the deck)
    - cards played are cast away, meaning the game ends when all cards have been played
    At every turn, you can either put down a card or end the play, if you are the first player of the current turn.
    Therefore, there are only 2 actions possible.

    Action table:
    index   | action name   |  extra_info              |
    ----------------------------------------------------
    0       | put card down | card suite and symbol    |
    1       | end turn      | nothing                  |

    Reward system:
    - + amount of points earned at the end of the play
    - - amount of points opponent has -"-
    - +5 points for winning
    - -5 points for losing
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.seed()

        self.deck = build_deck()
        self.player_hand = []
        self.adversary_hand = []
        self.played_cards = []
        self.used_cards = []
        self.is_first_player = True
        self.is_challenging = False

        self.player_points = 0
        self.adversary_points = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_winning(self):
        if self.is_first_player and not self.is_challenging:
            return True
        if not self.is_first_player and self.is_challenging:
            return True
        return False

    def _get_obs(self):
        return self.player_hand, self.played_cards[0] if len(self.played_cards) > 0 else None, \
               self.played_cards + self.used_cards, play_value(self.played_cards), \
               self.is_first_player, self.is_winning()

    def check_legal_put(self, card):
        if not self.is_challenging:
            # when there's no challenge going on, any card goes
            return not self.is_first_player or (self.is_first_player and len(self.played_cards) == 0)
        if not self.is_first_player:
            # when there's a challenge going on, every player but the first player can put down anything
            return True
        return card[0] == "7" or card[0] == self.played_cards[0][0]

    def check_start_challenge(self, card):
        return len(self.played_cards) > 0 and (card[0] == "7" or card[0] == self.played_cards[0][0])

    def build_state_from_env(self):
        game = SepticaMinmax(deepc(self.player_hand), deepc(self.adversary_hand), deepc(self.played_cards), deepc(self.deck),
                             self.is_challenging, SepticaMinmax.MINP if self.is_first_player else SepticaMinmax.MAXP,
                             self.player_points, self.adversary_points, copy.deepcopy(self.np_random), 0)
        return SepticaState(game_state=game, current_player=SepticaMinmax.MAXP, depth=4)

    def step(self, action, extra_info=None):
        reward = 0
        if action == 0:
            # put down card
            assert extra_info is not None
            card = extra_info
            assert self.check_legal_put(card)
            self.player_hand.remove(card)
            self.played_cards.append(card)
            self.is_challenging = self.check_start_challenge(card)
        elif action == 1:
            assert len(self.played_cards) > 0  # can't end hand before playing
            # end hand
            if self.is_challenging:
                self.adversary_points += play_value(self.played_cards)
                reward -= play_value(self.played_cards)
            else:
                self.player_points += play_value(self.played_cards)
                reward += play_value(self.played_cards)
            self.used_cards.extend(self.played_cards)
            self.played_cards = []
            if len(self.deck) > 0:
                self.player_hand, self.adversary_hand, self.deck = draw_until(self.deck, self.player_hand, self.adversary_hand, 4, self.np_random)
            self.is_first_player = not self.is_first_player
            self.is_challenging = False
        done = 0
        if len(self.deck) == 0 and len(self.player_hand) == 0 and len(self.played_cards) == 0:
            # reward += 5 * (1 if self.player_points > self.adversary_points else -1) # might have no effect
            done = 1

        # adversary stuff
        if not done:
            next_state = alpha_beta(state=self.build_state_from_env()).best_next_state
            print(f"score is {next_state.score}")
            game_state = next_state.game_state  # type: SepticaMinmax
            self.player_hand = game_state.player_hand
            self.adversary_hand = game_state.adversary_hand
            if len(game_state.adversary_hand) == 0:
                self.used_cards.extend(self.played_cards)
            self.played_cards = game_state.played_cards
            self.deck = game_state.deck
            self.player_points = game_state.player_score
            self.adversary_points = game_state.adversary_score
            self.is_challenging = game_state.is_challenging
            self.is_first_player = game_state.first_player == SepticaMinmax.MINP

        done = 0
        if len(self.deck) == 0 and len(self.player_hand) == 0 and len(self.played_cards) == 0:
            # reward += 5 * (1 if self.player_points > self.adversary_points else -1) # might have no effect
            done = 1
        return self._get_obs(), reward, done

    def reset(self):
        self.deck = build_deck()
        self.player_hand, self.deck = draw_hand(self.deck, self.np_random)
        self.adversary_hand, self.deck = draw_hand(self.deck, self.np_random)
        self.played_cards = []  # the player always begins the match
        self.used_cards = []
        self.is_challenging = False

        self.player_points = 0
        self.adversary_points = 0
        return self._get_obs()

    def render(self, mode="human"):
        print(f"Your hand is {self.player_hand}.")
        print(f"Adversary has {len(self.adversary_hand)} more cards.")
        print(f"Dealer has {self.adversary_hand} cards.")
        print(f"Cards down are {self.played_cards}.")
        print(f"Your score is {self.player_points}.")
        print(f"Adv score is {self.adversary_points}")
        action = None
        extra_info = None
        action = input("State your action (with extra info if req):")
        action = action.split(" ")
        if len(action) >= 2:
            extra_info = " ".join(action[1:])
        action = int(action[0])
        _, reward, done = self.step(action, extra_info)
        print(f"Pot after adv move: {self.played_cards}.")
        print(f"You got a reward of {reward}.")
        print("-"*100)
        if done:
            print(f"Game done.")
        return done

if __name__ == "__main__":
    env = SepticaEnv()
    env.reset()

    while not env.render():
        continue