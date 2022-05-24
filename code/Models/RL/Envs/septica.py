import copy

import gym
from gym import spaces
from gym.utils import seeding

from Models.RL.Envs.septica_utils import draw_card, draw_hand, build_deck, play_value, draw_until, shuffle_deck, REWARD_MULT
from Models.RL.Envs.septica_minmax import SepticaMinmax, alpha_beta, SepticaState
import Models.RL.septica_agent as sa


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
        self.used_cards = set()
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
               self.used_cards, play_value(self.played_cards), \
               self.is_first_player, self.is_challenging

    def _get_adv_obs(self):
        return self.adversary_hand, self.played_cards[0] if len(self.played_cards) > 0 else None, \
               self.used_cards, play_value(self.played_cards), \
               not self.is_first_player, self.is_challenging or len(self.played_cards) == 1

    def check_legal_put(self, card):
        if not self.is_challenging:
            # when there's no challenge going on, any card goes
            return not self.is_first_player or (self.is_first_player and len(self.played_cards) == 0)
        if not self.is_first_player and len(self.played_cards) > 0:
            # when there's a challenge going on, every player but the first player can put down anything
            return True
        return card[0] == "7" or card[0] == self.played_cards[0][0]

    def check_start_challenge(self, card):
        return len(self.played_cards) > 0 and (card[0] == "7" or card[0] == self.played_cards[0][0] or len(self.played_cards) == 1)

    def build_state_from_env(self):
        game = SepticaMinmax(deepc(self.player_hand), deepc(self.adversary_hand), deepc(self.played_cards), deepc(self.deck),
                             self.is_challenging, int(not self.is_first_player),
                             self.player_points, self.adversary_points, deepc(self.np_random), 0)
        return SepticaState(game_state=game, current_player=SepticaMinmax.MAXP, depth=4)

    def action_processing(self, action, extra_info=None):
        reward = 0
        if action == 0:
            # put down card
            assert extra_info is not None
            card = extra_info
            assert self.check_legal_put(card)
            if card[0] == "7" and play_value(self.played_cards) == 0:
                # discourage from wasting 7s
                reward -= 0.5 * REWARD_MULT
            self.player_hand.remove(card)
            self.played_cards.append(card)
            self.used_cards.update(self.played_cards)
            self.is_challenging = self.check_start_challenge(card)
        elif action == 1:
            # can only come here if they are the first player
            assert self.is_first_player
            assert len(self.played_cards) > 0  # can't end hand before playing
            # end hand
            if self.is_challenging:
                won = False
                self.adversary_points += play_value(self.played_cards) * REWARD_MULT
                reward -= play_value(self.played_cards) * REWARD_MULT
            else:
                won = True
                self.player_points += play_value(self.played_cards) * REWARD_MULT
                reward += play_value(self.played_cards) * REWARD_MULT
            self.used_cards.update(self.played_cards)
            self.played_cards = []
            if len(self.deck) > 0:
                self.player_hand, self.adversary_hand, self.deck = draw_until(self.deck, self.player_hand, self.adversary_hand, 4, self.np_random)
            self.is_first_player = won
            self.is_challenging = False
        return reward

    def step(self, action, extra_info=None):
        if not (len(self.played_cards) == 0 and not self.is_first_player):
            reward = self.action_processing(action, extra_info)
        else:
            reward = 0

        done = 0
        if len(self.deck) == 0 and len(self.player_hand) == 0 and len(self.played_cards) == 0:
            reward += 5 * (1 if self.player_points > self.adversary_points else -1) * REWARD_MULT  # might have no effect
            done = 1
        if not done and self.player_points + self.adversary_points == 8 * REWARD_MULT:
            done = 1
            reward += 5 * (
                1 if self.player_points > self.adversary_points else -1) * REWARD_MULT  # might have no effect

        # adversary stuff
        if not done and not (len(self.played_cards) == 0 and self.is_first_player):
            next_state = alpha_beta(state=self.build_state_from_env()).best_next_state
            game_state = next_state.game_state  # type: SepticaMinmax
            self.player_hand = game_state.player_hand
            self.adversary_hand = game_state.adversary_hand
            self.used_cards.update(game_state.played_cards)
            self.played_cards = game_state.played_cards
            self.deck = game_state.deck
            self.player_points = game_state.player_score
            self.adversary_points = game_state.adversary_score
            self.is_challenging = game_state.is_challenging
            self.is_first_player = game_state.first_player == SepticaMinmax.MINP
            reward += game_state.reward

        if not done and len(self.deck) == 0 and len(self.player_hand) == 0 and len(self.played_cards) == 0:
            reward += 5 * (1 if self.player_points > self.adversary_points else -1) * REWARD_MULT # might have no effect
            done = 1
        if not done and self.player_points + self.adversary_points == 8 * REWARD_MULT:
            done = 1
            reward += 5 * (1 if self.player_points > self.adversary_points else -1) * REWARD_MULT # might have no effect

        return self._get_obs(), reward, done

    def reset(self):
        self.deck = shuffle_deck(build_deck())
        self.player_hand, self.deck = draw_hand(self.deck, self.np_random)
        self.adversary_hand, self.deck = draw_hand(self.deck, self.np_random)
        self.played_cards = []  # the player always begins the match
        self.used_cards = set()
        self.is_challenging = False
        self.is_first_player = False

        self.player_points = 0
        self.adversary_points = 0
        return self._get_obs()

    def step_agent(self, action, extra_info=None):
        if not (len(self.played_cards) == 0 and not self.is_first_player):
            reward = self.action_processing(action, extra_info)
        else:
            reward = 0
        done = 0
        if len(self.deck) == 0 and len(self.player_hand) == 0 and len(self.played_cards) == 0:
            # reward += 5 * (1 if self.player_points > self.adversary_points else -1) # might have no effect
            done = 1

        if not done and not (len(self.played_cards) == 0 and self.is_first_player):
            agent = sa.get_septica_agent(self)
            # agent = None
            print(self._get_adv_obs())
            action = agent.get_action([agent.process_state(self._get_adv_obs())], eps=0)[0]
            action, extra_info = action
            assert action is None or self.action_space.contains(action)
            if action == 0:
                # put down card
                assert extra_info is not None
                card = extra_info
                # assert self.check_legal_put(card)
                self.adversary_hand.remove(card)
                self.played_cards.append(card)
                self.used_cards.update(self.played_cards)
                self.is_challenging = self.check_start_challenge(card)
            elif action == 1:
                assert not self.is_first_player
                assert len(self.played_cards) > 0  # can't end hand before playing
                # end hand
                if self.is_challenging:
                    won = True
                    self.player_points += play_value(self.played_cards) * REWARD_MULT
                    reward += play_value(self.played_cards) * REWARD_MULT
                else:
                    won = False
                    self.adversary_points += play_value(self.played_cards) * REWARD_MULT
                    reward -= play_value(self.played_cards) * REWARD_MULT
                self.used_cards.update(self.played_cards)
                self.played_cards = []
                if len(self.deck) > 0:
                    self.player_hand, self.adversary_hand, self.deck = draw_until(self.deck, self.player_hand,
                                                                                  self.adversary_hand, 4,
                                                                                  self.np_random)
                self.is_first_player = won
                self.is_challenging = False
            print(self._get_obs())

        done = 0
        if len(self.deck) == 0 and len(self.player_hand) == 0 and len(self.played_cards) == 0:
            reward += 5 * (1 if self.player_points > self.adversary_points else -1) * REWARD_MULT  # might have no effect
            done = 1

        return self._get_obs(), reward, done

    def render(self, mode="human"):
        print(f"Your hand is {self.player_hand}.")
        print(f"Adversary has {len(self.adversary_hand)} more cards.")
        # print(f"Dealer has {self.adversary_hand} cards.")
        print(f"Cards down are {self.played_cards}.")
        print(f"Your score is {self.player_points}.")
        print(f"Adv score is {self.adversary_points}")
        action = None
        extra_info = None
        if not (len(self.played_cards) == 0 and not self.is_first_player):
            action = input("State your action (with extra info if req):")
            action = action.split(" ")
            if len(action) >= 2:
                extra_info = " ".join(action[1:])
        else:
            action = "0"
            extra_info = None
        action = int(action[0])
        _, reward, done = self.step_agent(action, extra_info)
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