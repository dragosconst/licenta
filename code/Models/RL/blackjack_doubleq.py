# blackjack without card counting
import random
import copy
import dill as pickle
from collections import defaultdict
from typing import Dict

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
        action = agent.select_action(state)
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


class QAgent:
    def __init__(self, env, lr=0.1, gamma=1.0,
                 start_epsilon=1.0, end_epsilon=0.05, epsilon_decay=0.99999):
        self.env = env
        self.n_action = self.env.action_space.n
        self.policy = defaultdict(lambda: 0)  # always stay as default init policy
        self.gamma = gamma
        self.lr = lr

        # action values
        # Q(St, At)
        self.q1 = defaultdict(lambda: np.zeros(self.n_action))  # action value
        self.q2 = defaultdict(lambda: np.zeros(self.n_action))  # action value
        self.flip = 1

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    # get epsilon to use for each episode using epsilon decay
    def get_epsilon(self, n_episode):
        return max(self.start_epsilon * (self.epsilon_decay ** (n_episode / 7)), self.end_epsilon)

    # select action based on epsilon greedy (or  not)
    def select_action(self, state, epsilon=None):
        split, *_ = state
        if epsilon is not None and np.random.rand() < epsilon:
            pos_actions = [action for action in range(self.n_action)]
            if split is None:
                pos_actions.remove(2)
            action = random.sample(pos_actions, 1)[0]
        else:
            if state in self.q1 or state in self.q2:
                action = np.argmax([max(a1, a2) for a1, a2 in zip(self.q1[state], self.q2[state])])
            else:
                pos_actions = [action for action in range(self.n_action)]
                if split is None:
                    pos_actions.remove(2)
                action = random.sample(pos_actions, 1)[0]
        return action

    def update_q_values(self, reward, state, action, next_state, done):
        self.flip = random.randint(1, 3)
        split, *_ = next_state
        if self.flip == 1:
            max_action = None
            max_q = -100
            for idx, q1_action in enumerate(self.q1[next_state]):
                if split is None and idx == 2:
                    continue
                if q1_action > max_q:
                    max_q = q1_action
                    max_action = idx
            self.q1[state][action] = self.q1[state][action] + self.lr * (reward + self.gamma *
                                                                         self.q2[next_state][max_action] * (1 - done) -
                                                                         self.q1[state][action])
        else:
            max_action = None
            max_q = -100
            for idx, q2_action in enumerate(self.q2[next_state]):
                if split is None and idx == 2:
                    continue
                if q2_action > max_q:
                    max_q = q2_action
                    max_action = idx
            self.q2[state][action] = self.q2[state][action] + self.lr * (reward + self.gamma *
                                                                         self.q1[next_state][max_action] * (1 - done) -
                                                                         self.q2[state][action])

    # run episode with current policy
    def run_episode(self, eps=None, env=None):
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
            action = self.select_action(state, eps)
            output = local_env.step(action)
            if len(output) == 2:
                e1, e2 = output
                next_state_e1 = e1._get_obs()
                next_state_e2 = e2._get_obs()
                self.update_q_values(0, state, action, next_state_e1, done)
                self.update_q_values(0, state, action, next_state_e2, done)
                self.run_episode(eps, e1)
                self.run_episode(eps, e2)
                done = True
                continue
            next_state, reward, done, info = output
            self.update_q_values(reward, state, action, next_state, done)
            state = next_state

    # print rough estimation of current performance of agent
    def print_value(self, epoch, print_freq, matches_num):
        if epoch > 0 and epoch % print_freq == 0:
            samples = matches_num
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
            with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_doubleq_fixed.model",
                      "wb") as f:
                pickle.dump(self, f)

    def train(self, num_epochs):
        for e in trange(num_epochs):
            eps = self.get_epsilon(e)
            self.env.reset()
            self.run_episode(eps, self.env)
            self.print_value(e, 100 * 10 ** 3, 10 ** 5)

if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    env2 = BlackjackEnvSplit(natural=True)

    bj_agent = QAgent(env2)
    # bj_agent.train(num_epochs=int(3.5 * 10 ** 6))
    # with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_doubleq_fixed.model", "wb") as f:
    #     pickle.dump(bj_agent, f)
    with open("D:\\facultate stuff\\licenta\\data\\rl_models\\bj_doubleq_fixed.model", "rb") as f:
        bj_agent = pickle.load(f)

    # --------------------------------
    # draw moves table
    # --------------------------------
    NO_SPLITS = 28
    SPLITS = 36
    moves = np.zeros((SPLITS, 10))
    moves -= 1
    q = set(bj_agent.q1.keys())
    q.update(set(bj_agent.q2.keys()))
    for state in q:
        split, hand, dealer, ace = state
        action = np.argmax([max(a1, a2) for a1, a2 in zip(bj_agent.q1[state], bj_agent.q2[state])])
        if split is None:
            if action == 2:
                print(state)
            if not ace:
                if dealer > 1:
                    moves[hand - 5][dealer - 2] = action
                else:
                    moves[hand - 5][-1] = action
            else:
                if dealer > 1:
                    moves[hand - 13 + 17][dealer - 2] = action
                else:
                    moves[hand - 13 + 17][-1] = action
        else:
            if split == 1:
                split = 11
            if dealer > 1:
                moves[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][dealer - 2] = action
            else:
                moves[(10 if split in {"K", "Q", "J"} else int(split)) - 2 + 26][-1] = action

    # # --------------------------------
    # # heatmap of moves
    # # --------------------------------
    xlabels = [2, 3, 4, 5, 6, 7, 8, 9, 10, "A"]
    ylabs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, "A,2",
             "A,3", "A,4", "A,5", "A,6", "A,7", "A,8", "A,9", "A,10",
             "2,2", "3,3", "4,4", "5,5", "6,6", "7,7", "8,8", "9,9", "10,10", "A,A"]
    sn.heatmap(moves, xticklabels=xlabels, yticklabels=ylabs, linewidths=0.1, linecolor='gray')
    plt.savefig("doubleq.png")
    plt.show()

    samples = 5 * 10 ** 5
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