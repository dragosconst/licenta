# blackjack without card counting
import abc
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
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return epsilon

    def getAction(self, state):
        if state not in self.policy:
            return self.env.action_space.sample()
        return self.policy[state]

    # select action based on epsilon greedy (or  not)
    def select_action(self, state, epsilon):
        if epsilon is not None and np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            if state in self.q:
                action = self.policy[state]
            else:
                action = self.env.action_space.sample()
        return action

    # run episode with current policy
    def run_episode(self, eps=None):
        result = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state, eps)
            next_state, reward, done, info = self.env.step(action)
            result.append((state, action, reward, next_state, done))
            state = next_state
        return result

    # policy update with both e-greedy and regular argmax greedy
    def update_policy_q(self, eps=None):
        for state, values in self.q.items():
            if eps is not None: # e-Greedy policy updates
                if np.random.rand() < eps:
                    self.policy[state] = self.env.action_space.sample() # sample a random action
                else:
                    self.policy[state] = np.argmax(values)
            else: # Greedy policy updates
                self.policy[state] = np.argmax(values)

    # mc control GLIE
    # using something similar to every-visit mc
    def mc_control(self, n_episode=10, first_visit=True):
        for t in trange(n_episode):
            # Get an epsilon for this episode - used in e-greedy policy update
            eps = self.get_epsilon(t)

            # Generate an episode following current policy
            transitions = self.run_episode()

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

class TablePlayer():
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
        return self.action_table[player * 10 + dealer]

if __name__ == "__main__":
    def run_single_episode(env, agent):
        result = []
        state = env.reset()
        done = False
        while not done:
            action = agent.getAction(state)
            next_state, reward, done, info = env.step(action)
            result.append((state, action, reward, next_state, done))
            state = next_state
        return result  # must return a list of tuples (state,action,reward,next_state,done)

    env = gym.make('Blackjack-v1')

    # MC eval
    a = MCAgent(env)
    # a.mc_control(n_episode=10 ** 6, first_visit=True)
    # with open("D:\\facultate stuff\\licenta\\data\\bj_firsvisit.model", "wb") as f:
    #     pickle.dump(a, f)
    with open("D:\\facultate stuff\\licenta\\data\\bj_firstvisit.model", "rb") as f:
        a = pickle.load(f)

    #--------------------------------
    # draw moves table
    #--------------------------------
    moves = np.zeros((28, 10))
    for state, action in a.policy.items():
        hand, dealer, ace = state
        if not ace:
            moves[hand-4][dealer-1] = action
        else:
            moves[hand-12+18][dealer-1] = action
    print("----|A|2|3|4|5|6|7|8|9|10|---")
    for idx, line in enumerate(moves):
        if idx <= 17:
            ph = idx + 4
        else:
            ph = "A," + str(idx-18).zfill(2)
        print(f"{ph}:".zfill(4), end="")
        for move in line:
            print(f"{int(move)}|", end="")
        print("---")

    #--------------------------------
    # heatmap of moves
    #--------------------------------
    xlabels = ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ylabs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, "A,A", "A,2",
             "A,3", "A,4", "A,5", "A,6", "A,7", "A,8", "A,9", "A,10"]
    sn.heatmap(moves, xticklabels=xlabels, yticklabels=ylabs,linewidths=0.1, linecolor='gray')
    plt.savefig("firstvisit_moves.png")
    plt.show()
    # print(moves)

    # a = TablePlayer()
    samples = 10 ** 6
    good = 0
    for epIndex in trange(samples):
        transitions = run_single_episode(env, a)
        if transitions[-1][2] >= 1:
            good += 1
    print(f"win rate of {(good/samples)*100}%")