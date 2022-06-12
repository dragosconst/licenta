# blackjack without card counting
import random
import copy
import dill as pickle
from collections import defaultdict

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sn

from Models.RL.Envs.blackjack_splitting import BlackjackEnvSplit


def run_single_episode(env, agent, state=None):
    result = []

    if state is None:
        state = env.reset()
    done = False
    goodies = 0
    extra = 0
    while not done:
        action = agent.get_action(state)
        output = env.step(action)
        if len(output) == 2:
            e1, e2 = output
            r1, add1, e = run_single_episode(env=e1, agent=agent, state=e1._get_obs())
            goodies += add1
            extra += e
            if r1[-1][2] >= 1:
                goodies += 1
            r2, add2, e = run_single_episode(env=e2, agent=agent, state=e2._get_obs())
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


class MCAgent:
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
        self.action_visits = defaultdict(
            lambda: np.zeros(self.n_action))  # number of actions by type performed in each state
        self.state_visits = defaultdict(lambda: 0)

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    # get epsilon to use for each episode using epsilon decay
    def get_epsilon(self, n_episode):
        return max(self.start_epsilon * (self.epsilon_decay ** (n_episode / 7)), self.end_epsilon)

    def get_epsilon_visit_based(self, state):
        eps_decay = 0.999
        n = self.state_visits[state]
        return max(self.start_epsilon * (eps_decay ** n), self.end_epsilon)

    def get_action(self, state):
        split, *_ = state
        if state not in self.policy:
            pos_actions = [action for action in range(self.n_action)]
            if split is None:
                pos_actions.remove(2)
            action = random.sample(pos_actions, 1)[0]
            return action
        return self.policy[state]

    # select action based on epsilon greedy (or  not)
    def select_action(self, state, epsilon):
        split, *_ = state
        if epsilon is not None and np.random.rand() < epsilon:
            pos_actions = [action for action in range(self.n_action)]
            if split is None:
                pos_actions.remove(2)
            action = random.sample(pos_actions, 1)[0]
        else:
            if state in self.q:
                action = self.policy[state]
            else:
                pos_actions = [action for action in range(self.n_action)]
                if split is None:
                    pos_actions.remove(2)
                action = random.sample(pos_actions, 1)[0]
        return action

    # run episode with current policy
    def run_episode(self, eps=None, env=None, first_visit=True):
        result = []
        local_env = self.env
        if env is None:
            state = local_env.reset()
        else:
            local_env = env
            state = local_env._get_obs()
        ini_env = copy.deepcopy(local_env)
        done = False
        while not done:
            action = self.select_action(state, None)
            output = local_env.step(action)
            if len(output) == 2:
                e1, e2 = output
                r1, _ = self.run_episode(eps, e1, first_visit=first_visit)
                self.mc_control_one_ep(r1, eps, first_visit)
                r2, _ = self.run_episode(eps, e2, first_visit=first_visit)
                self.mc_control_one_ep(r2, eps, first_visit)
                done = True
                result.append((state, action, r1[-1][2] + r2[-1][2], None, done))
                continue
            next_state, reward, done, info = output
            result.append((state, action, reward, next_state, done))
            state = next_state
        return result, ini_env

    # policy update with both e-greedy and regular argmax greedy
    def update_policy_q(self, eps=None):
        for state, values in self.q.items():
            split, *_ = state
            if eps is not None:  # e-Greedy policy updates
                if np.random.rand() < eps:
                    pos_actions = [action for action in range(self.n_action)]
                    if split is None:
                        pos_actions.remove(2)
                    action = random.sample(pos_actions, 1)[0]
                    self.policy[state] = action
                else:
                    self.policy[state] = np.argmax(values)
            else:  # Greedy policy updates
                self.policy[state] = np.argmax(values)

    # used for splitting
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
            self.state_visits[st] += 1
            self.q[st][at] = self.q[st][at] + (1 / self.action_visits[st][at]) * (G - self.q[st][at])

        self.update_policy_q(eps)

    # print rough estimation of current performance of agent
    def print_value(self, epoch, print_freq, matches):
        if epoch > 0 and epoch % print_freq == 0:
            samples = matches
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
            print(f"Win rate so far is {(good / (samples + extra)) * 100}%, epoch {epoch}.")
            with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_BIG_new_action_space_fixed.model",
                      "wb") as f:
                pickle.dump(self, f)

    # mc control GLIE
    # using something similar to every-visit mc
    def mc_control(self, n_episode=10, first_visit=True):
        for e in trange(n_episode):
            # Get an epsilon for this episode - used in e-greedy policy update
            eps = self.get_epsilon(e)

            # Generate an episode following current policy
            transitions, old_env = self.run_episode(eps=eps, first_visit=first_visit)

            states, *_ = zip(*transitions)
            states_replay = 0
            if states[0][0] is not None and (int(states[0][0]) != 10 or int(
                    states[0][2]) != 10):  # first state was a split? in which case, do a state replay 3-4 times
                states_replay = 3
            if states[0][-2] and (int(states[0][1]) != 21 or int(states[0][2]) != 10):
                states_replay = 3

            while states_replay >= 0:
                self.mc_control_one_ep(transitions, eps, first_visit)
                if states_replay == 0:
                    self.print_value(e, 100 * 10 ** 3, 10 ** 5)

                states_replay -= 1
                if states_replay > 0:
                    transitions, old_env = self.run_episode(eps=eps, first_visit=first_visit, env=old_env)


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

    def get_action(self, state):
        player, dealer, ace = state
        if player > 17:
            return 0
        if not ace:
            player = self.hard_mappings[player]
        else:
            if player == 12:
                return 0
            player = self.soft_mappings[player]
        return self.action_table[player * 10 + (dealer - 2 if dealer > 1 else 9)]


class TablePlayerSplits():
    def __init__(self):
        self.state = None
        # table from https://images-na.ssl-images-amazon.com/images/I/81NSUO6ffyL.jpg
        self.action_table = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # ---regular scores
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
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # ---aces
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # ---splits
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

    def get_action(self, state):
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
        return self.action_table[player * 10 + (dealer - 2 if dealer > 1 else 9)]


class TablePlayerFull():
    def __init__(self):
        self.state = None
        # table from https://images-na.ssl-images-amazon.com/images/I/81NSUO6ffyL.jpg
        self.action_table = [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # ---regular scores
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 8
            1, 3, 3, 3, 3, 1, 1, 1, 1, 1,  # 9
            3, 3, 3, 3, 3, 3, 3, 3, 1, 1,  # 10
            3, 3, 3, 3, 3, 3, 3, 3, 3, 1,  # 11
            1, 1, 0, 0, 0, 1, 1, 1, 1, 1,  # 12
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 13
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 14
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 15
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 16
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 17+
            1, 1, 1, 3, 3, 1, 1, 1, 1, 1,  # ---aces
            1, 1, 1, 3, 3, 1, 1, 1, 1, 1,  # 3
            1, 1, 3, 3, 3, 1, 1, 1, 1, 1,  # 4
            1, 1, 3, 3, 3, 1, 1, 1, 1, 1,  # 5
            1, 3, 3, 3, 3, 1, 1, 1, 1, 1,  # 6
            0, 3, 3, 3, 3, 0, 0, 1, 1, 1,  # 7
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 8
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 9
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # ---splits;A and 8
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10
            2, 2, 2, 2, 2, 0, 2, 2, 0, 0,  # 9
            2, 2, 2, 2, 2, 2, 1, 1, 1, 1,  # 7
            1, 2, 2, 2, 2, 1, 1, 1, 1, 1,  # 6
            3, 3, 3, 3, 3, 3, 3, 3, 1, 1,  # 5
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 4
            1, 1, 2, 2, 2, 2, 1, 1, 1, 1  # 2,3
        ]
        self.hard_mappings = {5: 0, 6: 0, 7: 0, 8: 1, 9: 2, 10: 3,
                              11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9,
                              17: 10, 18: 10, 19: 10, 20: 10, 21: 10}
        self.soft_mappings = {3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16,
                              9: 17, 10: 18, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16,
                              19: 17, 20: 18}
        self.splits_mappings = {1: 19, 8: 19, 10: 20, 9: 21, 7: 22, 6: 23, 5: 24, 4: 25, 3: 26, 2: 26}
        self.dealer_mappings = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 1: 8}

    def get_action(self, state):
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
        return self.action_table[player * 10 + (dealer - 2 if dealer > 1 else 9)]


class TablePlayerNewRules():
    def __init__(self):
        self.state = None
        # table from https://wizardofodds.com/blackjack/images/bj_2d_h17.gif
        # I will define action 4 as double or hit and action 5 as double or stand
        # surrenders are ignored
        self.action_table = [
        #   2, 3, 4, 5, 6, 7, 8, 9, 10, A
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # ---regular scores
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 8
            3, 3, 3, 3, 3, 3, 1, 1, 1, 1,  # 9
            3, 3, 3, 3, 3, 3, 3, 3, 3, 1,  # 10
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 11
            1, 1, 0, 0, 0, 1, 1, 1, 1, 1,  # 12
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 13
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 14
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 15
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  # 16
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 17+
            1, 1, 1, 3, 3, 1, 1, 1, 1, 1,  # ---aces, 2
            1, 1, 3, 3, 3, 1, 1, 1, 1, 1,  # 3
            1, 1, 3, 3, 3, 1, 1, 1, 1, 1,  # 4
            1, 1, 3, 3, 3, 1, 1, 1, 1, 1,  # 5
            1, 3, 3, 3, 3, 1, 1, 1, 1, 1,  # 6
            4, 4, 4, 4, 4, 0, 0, 1, 1, 1,  # 7
            0, 0, 0, 0, 4, 0, 0, 0, 0, 0,  # 8
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 9
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # ---splits;A and 8
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 10
            2, 2, 2, 2, 2, 0, 2, 2, 0, 0,  # 9
            2, 2, 2, 2, 2, 2, 1, 1, 1, 1,  # 7
            2, 2, 2, 2, 2, 1, 1, 1, 1, 1,  # 6
            3, 3, 3, 3, 3, 3, 3, 3, 1, 1,  # 5
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 4
            1, 1, 2, 2, 2, 2, 1, 1, 1, 1  # 2,3
        ]
        self.hard_mappings = {5: 0, 6: 0, 7: 0, 8: 1, 9: 2, 10: 3,
                              11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9,
                              17: 10, 18: 10, 19: 10, 20: 10, 21: 10}
        self.soft_mappings = {3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16,
                              9: 17, 10: 18, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16,
                              19: 17, 20: 18}
        self.splits_mappings = {1: 19, 8: 19, 10: 20, 9: 21, 7: 22, 6: 23, 5: 24, 4: 25, 3: 26, 2: 26}
        self.dealer_mappings = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 9: 6, 10: 7, 1: 8}

    def get_action(self, state):
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
        action = self.action_table[player * 10 + (dealer - 2 if dealer > 1 else 9)]
        return action

if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    env2 = BlackjackEnvSplit(natural=True)

    # MC eval
    bj_agent = MCAgent(env2)
    # bj_agent.mc_control(n_episode=int(3.5 * 10 ** 6), first_visit=True)
    # with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_BIG_new_action_space_fixed.model", "wb") as f:
    #     pickle.dump(bj_agent, f)
    with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_firstvisit_BIG_new_action_space_fixed.model", "rb") as f:
        bj_agent = pickle.load(f)

    # --------------------------------
    # draw moves table
    # --------------------------------
    NO_SPLITS = 28
    SPLITS = 36
    moves = np.zeros((SPLITS, 10))
    moves_visits = np.zeros((SPLITS, 10))
    moves -= 1
    for state, action in bj_agent.policy.items():
        split, hand, dealer, ace = state
        if split is None:
            if action == 2:
                print(state)
            if not ace:
                if dealer > 1:
                    moves[hand - 5][dealer - 2] = action
                    moves_visits[hand - 5][dealer - 2] = bj_agent.action_visits[state][action]
                else:
                    moves[hand - 5][-1] = action
                    moves_visits[hand - 5][-1] = bj_agent.action_visits[state][action]
            else:
                if dealer > 1:
                    moves[hand - 13 + 17][dealer - 2] = action
                    moves_visits[hand - 13 + 17][dealer - 2] = bj_agent.action_visits[state][action]
                else:
                    moves[hand - 13 + 17][-1] = action
                    moves_visits[hand - 13 + 17][-1] = bj_agent.action_visits[state][action]
        else:
            if split == 1:
                split = 11
            if dealer > 1:
                moves[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][dealer - 2] = action
                moves_visits[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][dealer - 2] = \
                    bj_agent.action_visits[state][action]
            else:
                moves[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][-1] = action
                moves_visits[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][-1] = \
                    bj_agent.action_visits[state][action]

    # # --------------------------------
    # # heatmap of moves
    # # --------------------------------
    xlabels = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]
    ylabs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, "A,2",
             "A,3", "A,4", "A,5", "A,6", "A,7", "A,8", "A,9", "A,10",
             "2,2", "3,3", "4,4", "5,5", "6,6", "7,7", "8,8", "9,9", "10,10", "A,A"]
    # sn.heatmap(moves, xticklabels=xlabels, yticklabels=ylabs, linewidths=0.1, linecolor='gray')
    # plt.savefig("firstvisit_moves_BIG_double_correct_state_replay.png")
    # plt.show()
    # sn.heatmap(moves_visits, xticklabels=xlabels, yticklabels=ylabs, linewidths=0.1, linecolor='gray', cmap="YlGnBu")
    # plt.savefig("firstvisit_moves_BIG_double_correct_state_replay_visits.png")
    # plt.show()
    print(moves)

    bj_agent = TablePlayerNewRules()
    samples = 1 * 10 ** 6
    extras = 0
    good = 0
    neutral = 0
    total_reward = 0
    for epIndex in trange(samples):
        transitions, add, extra = run_single_episode(env2, bj_agent)
        total_reward += transitions[-1][2]
        # good += add
        extras += extra
        # print(transitions[-1])
        if add > 0:
            good += add
            continue
        if transitions[-1][2] >= 1:
            good += 1
    print(f"win rate of {(good / (samples + extras)) * 100}%")
    print(f"Final reward is {total_reward}")
