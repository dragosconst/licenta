# blackjack without card counting
import random
import warnings
warnings.filterwarnings("ignore")

import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import dill as pickle
from tqdm import trange
import seaborn as sn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from Models.RL.Envs.blackjack_splitting import BlackjackEnvSplit


def run_single_episode(env, agent, state=None):
    result = []

    state = env.reset()
    done = False
    goodies = 0
    extra = 0
    while not done:
        action = agent.getAction(state)
        output = env.step(action)
        if len(output) == 2:
            e1, e2 = output
            r1, add1, e = run_single_episode(env=env, agent=agent, state=e1._get_obs())
            goodies += add1
            extra += e
            if r1[-1][2] >= 1:
                goodies += 1
            r2, add2, e = run_single_episode(env=env, agent=agent, state=e2._get_obs())
            if r2[-1][2] >= 1:
                goodies += 1
            goodies += add2
            extra += e
            done = True
            result.append((state, action, r1[-1][2] + r2[-1][2], None, done))
            continue
        next_state, reward, done, info = output
        result.append((state, action, reward, next_state, done))
        state = next_state
    return result, goodies, extra  # must return a list of tuples (state,action,reward,next_state,done)

class MCAgent():
    def __init__(self, env, gamma=1.0,
                 start_epsilon=1.0, end_epsilon=0.05, epsilon_decay=0.99999):

        self.env = env
        self.n_action = self.env.action_space.n
        self.policy = defaultdict(lambda: 0)  # always stay as default init policy
        self.gamma = gamma

        # action values
        # Q(St, At)
        self.q = defaultdict(lambda: np.zeros(self.n_action))  # action value

        # N(St, At)
        self.action_visits = defaultdict(lambda: np.zeros(self.n_action))  # number of actions by type performed in each state

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    # get epsilon to use for each episode using epsilon decay
    def get_epsilon(self, n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** (n_episode / 1.7)), self.end_epsilon)
        return epsilon

    def get_epsilon_visit_based(self, state, action):
        eps_decay = 0.9998
        n = self.action_visits[state][action]
        return max(self.start_epsilon * (eps_decay ** n), self.end_epsilon)

    def getAction(self, state):
        split, *_ = state
        if state not in self.policy:
            action = self.env.action_space.sample()
            while split is None and action == 2:
                action = self.env.action_space.sample()
            return action
        return self.policy[state]

    # select action based on epsilon greedy (or  not)
    def select_action(self, state, epsilon):
        split, *_ = state
        if epsilon is not None and np.random.rand() < epsilon:
            action = self.env.action_space.sample()
            while split is None and action == 2:
                action = self.env.action_space.sample()
        else:
            if state in self.q:
                action = self.policy[state]
            else:
                action = self.env.action_space.sample()
                while split is None and action == 2:
                    action = self.env.action_space.sample()
        return action

    # run episode with current policy
    def run_episode(self, eps=None, state=None, first_visit=True):
        result = []
        if state is None:
            state = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state, None)
            output = self.env.step(action)
            if len(output) == 2:
                e1, e2 = output
                r1 = self.run_episode(eps, state=e1._get_obs(), first_visit=first_visit)
                self.mc_control_one_ep(r1, eps, first_visit)
                r2 = self.run_episode(eps, state=e2._get_obs(), first_visit=first_visit)
                self.mc_control_one_ep(r2, eps, first_visit)
                done = True
                result.append((state, action, r1[-1][2] + r2[-1][2], None, done))
                continue
            next_state, reward, done, info = output
            result.append((state, action, reward, next_state, done))
            state = next_state
        return result

    # policy update with both e-greedy and regular argmax greedy
    def update_policy_q(self, eps=None):
        for state, values in self.q.items():
            split, *_ = state
            if eps is not None: # e-Greedy policy updates
                if np.random.rand() < eps:
                    self.policy[state] = self.env.action_space.sample() # sample a random action
                    while split is None and self.policy[state] == 2:
                        self.policy[state] = self.env.action_space.sample()
                else:
                    self.policy[state] = np.argmax(values)
            else: # Greedy policy updates
                self.policy[state] = np.argmax(values)


    def mc_control_one_ep(self, transitions, eps, first_visit=True):
        states, actions, rewards, _, _ = zip(*transitions)
        # create table of first visit timesteps for first visit MC
        if first_visit:
            first_visit_dict = {}
            for ts, s in enumerate(states):
                sa = (s, actions[ts])
                if sa not in first_visit_dict:
                    first_visit_dict[sa] = ts

        # Iterate over episode steps in reversed order, T-1, T-2, ....0
        G = 0  # return output
        for t in range(len(transitions) - 1, -1, -1):
            st = states[t]
            at = actions[t]

            G = self.gamma * G + rewards[t]
            if first_visit:
                if first_visit_dict[(st, at)] != t:
                    continue

            # this piece of code is great because it works for both every visit and first visit
            self.action_visits[st][at] += 1
            self.q[st][at] = self.q[st][at] + (1 / self.action_visits[st][at]) * (G - self.q[st][at])

        self.update_policy_q(eps)

    # mc control GLIE
    # using something similar to every-visit mc
    def mc_control(self, n_episode=10, first_visit=True):
        for e in trange(n_episode):
            # Get an epsilon for this episode - used in e-greedy policy update
            eps = self.get_epsilon(e)

            # Generate an episode following current policy
            transitions = self.run_episode(eps=eps, first_visit=first_visit)

            states, actions, rewards, _, _ = zip(*transitions)

            # create table of first visit timesteps for first visit MC
            if first_visit:
                first_visit_dict = {}
                for ts, s in enumerate(states):
                    sa = (s, actions[ts])
                    if sa not in first_visit_dict:
                        first_visit_dict[sa] = ts

            # Iterate over episode steps in reversed order, T-1, T-2, ....0
            G = 0 # return output
            for t in range(len(transitions)-1, -1, -1):
                st = states[t]
                at = actions[t]

                G = self.gamma * G + rewards[t]
                if first_visit:
                    if first_visit_dict[(st, at)] != t:
                        continue

                # this piece of code is great because it works for both every visit and first visit
                self.action_visits[st][at] += 1
                self.q[st][at] = self.q[st][at] + (1 / self.action_visits[st][at]) * (G - self.q[st][at])

            self.update_policy_q(eps)
            if e > 0 and e % (100 * 10 ** 3) == 0:
                samples = 10 ** 5
                good = 0
                extra = 0
                for epIndex in range(samples):
                    transitions, add, extras = run_single_episode(self.env, self)
                    extra += extras
                    if add > 0:
                        good += add
                        continue
                    # print(transitions[-1])
                    if transitions[-1][2] >= 1:
                        good += 1
                print(f"Win rate so far is {(good/(samples+extra))*100}%, epoch {e}.")
                with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_double_exp.model", "wb") as f:
                    pickle.dump(a, f)


class TablePlayerNoSplit():
    def __init__(self):
        self.state = None
        # table from https://images-na.ssl-images-amazon.com/images/I/81NSUO6ffyL.jpg
        self.action_table = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        self.hard_mappings = {4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 2, 10: 3,
                              11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9,
                              17: 10}
        self.soft_mappings = {3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16,
                              9: 17, 10: 18, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16,
                              19: 17, 20: 18}
        self.dealer_mappings = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 1: 8}

    def getAction(self, state):
        player, dealer, ace = state
        if player > 17:
            return 0
        if not ace:
            player = self.hard_mappings[player]
        else:
            if player == 12:
                return 0
            player = self.soft_mappings[player]
        return self.action_table[player * 10 + (dealer-2 if dealer > 1 else 9)]

class TablePlayerSplits():
    def __init__(self):
        self.state = None
        # table from https://images-na.ssl-images-amazon.com/images/I/81NSUO6ffyL.jpg
        self.action_table = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#---regular scores
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#---aces
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,#---splits
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 2, 0, 2, 2, 0, 0,
            2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
            1, 2, 2, 2, 2, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 2, 2, 2, 2, 1, 1, 1, 1
        ]
        self.hard_mappings = {5: 0, 6: 0, 7: 0, 8: 1, 9: 2, 10: 3,
                              11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9,
                              17: 10, 18: 10, 19: 10, 20: 10, 21: 10}
        self.soft_mappings = {3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16,
                              9: 17, 10: 18, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16,
                              19: 17, 20: 18}
        self.splits_mappings = {1: 19, 8: 19, 10: 20, 9: 21, 7: 22, 6: 23, 5: 24, 4: 25, 3: 26, 2: 26}
        self.dealer_mappings = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 1: 8}

    def getAction(self, state):
        split, player_sum, dealer, ace = state
        if player_sum == 21:
            return 0
        if not ace and split is None:
            player = self.hard_mappings[player_sum]
        elif ace and split is None:
            player = self.soft_mappings[player_sum]
        elif split is not None:
            split = 10 if split in {"K", "Q", "J"} else int(split)
            player = self.splits_mappings[split]
        return self.action_table[player * 10 + (dealer-2 if dealer > 1 else 9)]

class TablePlayerFull():
    def __init__(self):
        self.state = None
        # table from https://images-na.ssl-images-amazon.com/images/I/81NSUO6ffyL.jpg
        self.action_table = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#---regular scores
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#8
            1, 3, 3, 3, 3, 1, 1, 1, 1, 1,#9
            3, 3, 3, 3, 3, 3, 3, 3, 1, 1,#10
            3, 3, 3, 3, 3, 3, 3, 3, 3, 1,#11
            1, 1, 0, 0, 0, 1, 1, 1, 1, 1,#12
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,#13
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,#14
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,#15
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,#16
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,#17+
            1, 1, 1, 3, 3, 1, 1, 1, 1, 1,#---aces
            1, 1, 1, 3, 3, 1, 1, 1, 1, 1,#3
            1, 1, 3, 3, 3, 1, 1, 1, 1, 1,#4
            1, 1, 3, 3, 3, 1, 1, 1, 1, 1,#5
            1, 3, 3, 3, 3, 1, 1, 1, 1, 1,#6
            0, 3, 3, 3, 3, 0, 0, 1, 1, 1,#7
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,#8
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,#9
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,#---splits;A and 8
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,#10
            2, 2, 2, 2, 2, 0, 2, 2, 0, 0,#9
            2, 2, 2, 2, 2, 2, 1, 1, 1, 1,#7
            1, 2, 2, 2, 2, 1, 1, 1, 1, 1,#6
            3, 3, 3, 3, 3, 3, 3, 3, 1, 1,#5
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,#4
            1, 1, 2, 2, 2, 2, 1, 1, 1, 1#2,3
        ]
        self.hard_mappings = {5: 0, 6: 0, 7: 0, 8: 1, 9: 2, 10: 3,
                              11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9,
                              17: 10, 18: 10, 19: 10, 20: 10, 21: 10}
        self.soft_mappings = {3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16,
                              9: 17, 10: 18, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16,
                              19: 17, 20: 18}
        self.splits_mappings = {1: 19, 8: 19, 10: 20, 9: 21, 7: 22, 6: 23, 5: 24, 4: 25, 3: 26, 2: 26}
        self.dealer_mappings = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 1: 8}

    def getAction(self, state):
        split, player_sum, dealer, ace = state
        if player_sum == 21:
            return 0
        if not ace and split is None:
            player = self.hard_mappings[player_sum]
        elif ace and split is None:
            player = self.soft_mappings[player_sum]
        elif split is not None:
            split = 10 if split in {"K", "Q", "J"} else int(split)
            player = self.splits_mappings[split]
        return self.action_table[player * 10 + (dealer-2 if dealer > 1 else 9)]

if __name__ == "__main__":

    env = gym.make('Blackjack-v1')
    env2 = BlackjackEnvSplit(natural=True)

    # MC eval
    a = MCAgent(env2)
    # a.mc_control(n_episode=10 ** 6, first_visit=True)
    # with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_double_exp.model", "wb") as f:
    #     pickle.dump(a, f)
    with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_double_exp.model", "rb") as f:
        a = pickle.load(f)

    #--------------------------------
    # draw moves table
    #--------------------------------
    NO_SPLITS = 28
    SPLITS = 36
    moves = np.zeros((SPLITS, 10))
    moves_visits = np.zeros((SPLITS, 10))
    moves -= 1
    for state, action in a.policy.items():
        split, hand, dealer, ace = state
        if split is None:
            if action == 2:
                print(state)
            if not ace:
                if dealer > 1:
                    moves[hand - 5][dealer - 2] = action
                    moves_visits[hand - 5][dealer - 2] = a.action_visits[state][action]
                else:
                    moves[hand - 5][-1] = action
                    moves_visits[hand - 5][-1] = a.action_visits[state][action]
            else:
                if dealer > 1:
                    moves[hand - 13 + 17][dealer - 2] = action
                    moves_visits[hand - 13 + 17][dealer - 2] = a.action_visits[state][action]
                else:
                    moves[hand - 13 + 17][-1] = action
                    moves_visits[hand - 13 + 17][-1] = a.action_visits[state][action]
        else:
            if split == 1:
                split = 11
            if dealer > 1:
                moves[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][dealer - 2] = action
                moves_visits[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][dealer - 2] = a.action_visits[state][action]
            else:
                moves[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][-1] = action
                moves_visits[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][-1] = a.action_visits[state][action]

    # #--------------------------------
    # # heatmap of moves
    # #--------------------------------
    xlabels = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]
    ylabs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, "A,2",
             "A,3", "A,4", "A,5", "A,6", "A,7", "A,8", "A,9", "A,10",
             "2,2", "3,3", "4,4", "5,5", "6,6", "7,7", "8,8", "9,9", "10,10", "A,A"]
    sn.heatmap(moves, xticklabels=xlabels, yticklabels=ylabs,linewidths=0.1, linecolor='gray')
    plt.savefig("firstvisit_moves_doubles.png")
    plt.show()
    sn.heatmap(moves_visits, xticklabels=xlabels, yticklabels=ylabs,linewidths=0.1, linecolor='gray', cmap="YlGnBu")
    plt.savefig("firstvisit_moves_doubles_actionvisits.png")
    plt.show()
    # print(moves)

    # a = TablePlayerFull()
    samples = 5 * 10 ** 5
    extras = 0
    good = 0
    neutral = 0
    total_reward = 0
    for epIndex in trange(samples):
        transitions, add, extra = run_single_episode(env2, a)
        total_reward += transitions[-1][2]
        # good += add
        extras += extra
        # print(transitions[-1])
        if add > 0:
            good += add
            continue
        if transitions[-1][2] >= 1:
            good += 1
    print(f"win rate of {(good/(samples+extras))*100}%")
    print(f"Final reward is {total_reward}")