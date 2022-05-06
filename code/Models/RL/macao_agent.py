from typing import List, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import trange

from Models.RL.Envs.macao import MacaoEnv, same_suite
from Models.RL.Envs.macao_utils import build_deck
from Models.RL.macao_agent_utils import LinearScheduleEpsilon, ReplayBuffer


class MacaoModel(nn.Module):
    def __init__(self, device: str):
        super(MacaoModel, self).__init__()
        self.device = device

        self.hand_proj = nn.Linear(in_features=54, out_features=30)
        self.pot_proj = nn.Linear(in_features=54, out_features=30)
        self.suite_proj = nn.Linear(in_features=4, out_features=2)

        self.hidden_layers = nn.Sequential(
            nn.Linear(in_features=65, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=125),
            nn.ReLU(),
            nn.Linear(in_features=125, out_features=80),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(in_features=80, out_features=62)

    def forward(self, x):
        # x is tuple of (hand, cards_pot, player_turns, suites, just_put, deck_np), where each element is already preprocessed for the net
        # the preprocessing consists of turning them to their one hot encoding and summing them
        # and turning them to tensors
        hand, card_pot, player_turns, suites, just_put, deck_no = x
        hand = hand.to(self.device)
        card_pot = card_pot.to(self.device)
        player_turns = player_turns.to(self.device)
        suites = suites.to(self.device)
        just_put = just_put.to(self.device)
        deck_no = deck_no.to(self.device)

        hand_proj = self.hand_proj(hand)
        card_proj = self.pot_proj(card_pot)
        suites = self.suite_proj(suites)

        result = torch.cat((hand_proj, card_proj, player_turns, suites, just_put, deck_no))

        h = self.hidden_layers(result)
        output = self.output_layer(h)
        return output


class MacaoAgent:
    def __init__(self, env: MacaoEnv, replay_buff_size=100, gamma=0.9, batch_size=512, lr=1e-3, steps_per_dqn=20,
                 pre_train_steps=1, eps_scheduler=LinearScheduleEpsilon(), max_steps_per_episode: int=200):
        self.env = env
        self.full_deck = build_deck()
        self.suits = ["s", "c", "d", "h"]
        self.num_actions = self.env.action_space.n

        self.replay_buff_size = replay_buff_size
        self.replay_buffer = ReplayBuffer(buffer_size=self.replay_buff_size)
        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode

        self.lr = lr
        self.q1 = MacaoModel("cuda")
        # gradient clipping
        for p in self.q1.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        self.optim1 = Adam(self.q1.parameters(), lr=self.lr)
        self.q2 = MacaoModel("cuda")
        for p in self.q2.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        self.optim2 = Adam(self.q2.parameters(), lr=self.lr)
        self.batch_size = batch_size
        self.steps_per_dqn = steps_per_dqn
        self.pre_train_steps = pre_train_steps
        self.total_steps = 0
        self.eps_scheduler = eps_scheduler
        self.flip = 1
        self.total_steps = 0

    def step(self, action: int, extra_info: str=None) -> Tuple[Tuple, ...]:
        state, reward, done = self.env.step(action=action, extra_info=extra_info)
        return state, reward, done

    def process_state(self, state: Tuple) -> Tuple[torch.Tensor, ...]:
        """
        Change state to form expected by the model.
        """
        player_hand, last_cards, player_turns, suits, just_moved, deck = state
        new_hand = torch.zeros(len(self.full_deck))
        for card in player_hand:
            new_hand[self.full_deck.index(card)] = 1
        new_last = torch.zeros((len(self.full_deck)))
        for idx, card in enumerate(last_cards):
            if card is not None:
                new_last[self.full_deck.index(card)] = idx + 1 # encode information about order in cards pot
        new_suits = torch.zeros(4)
        suit_list = ["s", "c", "d", "h"]
        for suit in suits:
            new_suits[suit_list.index(suit)] = 1
        just_moved = torch.as_tensor([just_moved])

        return new_hand, new_last, torch.as_tensor(player_turns), new_suits, just_moved, torch.as_tensor(deck)

    def check_legal_action(self, state, action, extra_info):
        hand, cards_pot, player_turns, suits, just_moved, deck = state
        new_cards_pot = [None] * 5
        for idx, card in enumerate(cards_pot):
            if card:
                new_cards_pot[int(card.item()) - 1] = self.full_deck[idx]
        if action == 0:
            if player_turns > 0:
                return False
            last_card_idx = len(new_cards_pot) - 1
            # check if there's a drawing contest going on
            while just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and\
                                                    new_cards_pot[last_card_idx][0] == "5":
                # skip over redirects
                last_card_idx -= 1
            if just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and (
                    new_cards_pot[last_card_idx][0] in {"2", "3"} or new_cards_pot[last_card_idx][:3] \
                    == "jok"):
                # check if we are doing a valid contestation
                return extra_info[0] in {"2", "3", "4", "5"} or extra_info[:3] == "jok"

            # now check if we are in a waiting turns contest, i.e. aces
            if just_moved and new_cards_pot[last_card_idx] is not None and new_cards_pot[last_card_idx][0] == "A":
                return extra_info[0] == "A"

            # check for the special case of trying to put down a joker as a beginning of a contest
            if extra_info[:3] == "jok":
                if extra_info == "joker black":
                    return new_cards_pot[last_card_idx] is not None and new_cards_pot[last_card_idx][-1] in {"s", "c"}
                elif extra_info == "joker red":
                    return new_cards_pot[last_card_idx] is not None and new_cards_pot[last_card_idx][-1] in {"h", "d"}

            # finally, we are left with the case of trying to put a regular card over another regular card
            return new_cards_pot[last_card_idx] is not None and new_cards_pot[last_card_idx][0] == extra_info[0] or same_suite(suits, extra_info)
        elif action == 1:
            if player_turns > 0:
                return False
            if len([card for card in cards_pot if card is not None]) <= 1 and deck == 0:
                return False
            last_card_idx = len(new_cards_pot) - 1
            # check if there's a drawing contest going on
            while just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and\
                                                    new_cards_pot[last_card_idx][0] == "5":
                # skip over redirects
                last_card_idx -= 1
            if just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and (
                    new_cards_pot[last_card_idx][0] in {"2", "3"} or new_cards_pot[last_card_idx][:3] \
                    == "jok"):
                # can't pass during a contestation
                return False
            if just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and new_cards_pot[last_card_idx][0] == "A":
                return False
            return True
        elif action == 2:
            last_card_idx = len(new_cards_pot) - 1
            # check if there's a drawing contest going on
            while just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and\
                                                    new_cards_pot[last_card_idx][0] == "5":
                # skip over redirects
                last_card_idx -= 1
            return just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and (
                    new_cards_pot[last_card_idx][0] in {"2", "3"} or new_cards_pot[last_card_idx][:3] \
                    == "jok")
        elif action == 3:
            return just_moved and new_cards_pot[-1][0] == "A"
        elif action == 4:
            if player_turns > 0:
                return False
            last_card_idx = len(new_cards_pot) - 1
            # check if there's a drawing contest going on
            while just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and\
                                                    new_cards_pot[last_card_idx][0] == "5":
                # skip over redirects
                last_card_idx -= 1
            if just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and (
                    new_cards_pot[last_card_idx][0] in {"2", "3"} or new_cards_pot[last_card_idx][:3] \
                    == "jok"):
                # can't pass during a contestation
                return False
            if just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and new_cards_pot[last_card_idx][0] == "A":
                return False
            return True
        elif action == 5:
            last_card_idx = len(new_cards_pot) - 1
            # check if there's a drawing contest going on
            while just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and\
                                                    new_cards_pot[last_card_idx][0] == "5":
                # skip over redirects
                last_card_idx -= 1
            if just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and (
                    new_cards_pot[last_card_idx][0] in {"2", "3"} or new_cards_pot[last_card_idx][:3] \
                    == "jok"):
                # can't pass during a contestation
                return False
            if just_moved and last_card_idx >= 0 and new_cards_pot[last_card_idx] is not None and new_cards_pot[last_card_idx][0] == "A":
                return False
            return player_turns > 0
        return False # used for when it chooses action 4 but it has no 7s

    def idx_to_action(self, idx):
        """
        Function that converts an index from the output tensor of the dqn network to an action.
        The following mapping is used:
        - 0-53: for putting down xth card, with joker black and red being the last
        - 54: pass
        - 55: concede
        - 56: wait
        - 57-60: change suite
        - 61: don't do anything, used when waiting turns
        """
        if idx <= 53:
            return 0, self.full_deck[idx]
        elif idx == 54:
            return 1, None
        elif idx == 55:
            return 2, None
        elif idx == 56:
            return 3, None
        elif idx <= 60:
            return 4, self.suits[idx - 57]
        elif idx == 61:
            return 5, None

    def get_action(self, states, eps):
        p = random.random()

        if self.total_steps <= self.pre_train_steps or p < eps:
            actions_choice = np.random.choice(self.num_actions, size=len(states))
            actions = []
            for idx, action in enumerate(actions_choice):
                extra_info = None
                state = states[idx]
                hand = [self.full_deck[idx] for idx, one in enumerate(state[0]) if one == 1]
                if action == 0:
                    extra_info = random.choice(hand)  # randomly choose a card from hand
                elif action == 4:
                    valid = False
                    good_card = ""
                    for card in hand:
                        if card[0] == "7":
                            valid = True
                            good_card = card
                            break
                    action = action if valid else -1
                    extra_info = good_card + " " + random.choice(["s", "c", "h", "d"])
                while not self.check_legal_action(state, action, extra_info):
                    action = np.random.choice(self.num_actions)
                    extra_info = None
                    if action == 0:
                        extra_info = random.choice(hand)  # randomly choose a card from hand
                    elif action == 4:
                        valid = False
                        good_card = ""
                        for card in hand:
                            if card[0] == "7":
                                valid = True
                                good_card = card
                                break
                        action = action if valid else -1
                        extra_info = good_card + " " + random.choice(["s", "c", "h", "d"])
                actions.append((action, extra_info))
        else:
            if self.flip == 1:
                q_hat = self.q1(states)
            else:
                q_hat = self.q2(states)

            q_hat_sorted = np.argsort(q_hat, axis=1)
            actions = []
            for idx, action_idxs in enumerate(q_hat_sorted):
                for action_idx in reversed(action_idxs):
                    action, extra_info = self.idx_to_action(action_idx)
                    if self.check_legal_action(states[idx], action, extra_info):
                        actions.append((action, extra_info))
                        break
        return actions

    def run_episode(self):
        """
        Run a single training episode
        """
        init = self.env.reset()

        ep_reward = 0
        current_state = self.process_state(init)
        current_action = None
        total_loss = 0
        num_loss_comp = 0
        epsilon = 1.0
        for ep_step in trange(self.max_steps_per_episode):
            epsilon = self.eps_scheduler.get_value(step=self.total_steps)

            batch = [current_state]
            self.flip = random.choice([1, 2])
            actions = self.get_action(batch, epsilon)
            current_action, current_extra_info = actions[0]

            new_state, reward, done = self.step(current_action, current_extra_info)
            new_state_proc = self.process_state(new_state)
            ep_reward += reward

            self.replay_buffer.add(state=current_state, action=current_action, reward=reward, next_state=new_state_proc, done=done)

            if self.total_steps > self.pre_train_steps and len(self.replay_buffer) >= self.batch_size:
                loss = self.train_step()
                total_loss += loss
                num_loss_comp += 1

            current_state = new_state_proc
            self.total_steps += 1

            if done:
                break
        avg_loss = (total_loss/num_loss_comp if num_loss_comp else 0)
        return ep_reward, avg_loss, epsilon

    def train_step(self):
        train_batch = self.replay_buffer.sample(batch_size=self.batch_size)
        batch_states = [x[0] for x in train_batch]
        batch_actions = torch.as_tensor([x[1] for x in train_batch])
        batch_rewards = torch.as_tensor([x[2] for x in train_batch])
        batch_nexts =[x[3] for x in train_batch]
        batch_dones = torch.as_tensor([x[4] for x in train_batch])

        if self.flip == 1:
            # use q1 for next state computations
            q_hat = self.q1(batch_nexts)
        else:
            # use q2 for next state computations
            q_hat = self.q2(batch_nexts)

        q_hat_sorted = torch.argsort(q_hat, dim=1)
        actions = []
        actions_idx = []
        for idx, action_idxs in enumerate(q_hat_sorted):
            for action_idx in reversed(action_idxs):
                action, extra_info = self.idx_to_action(action_idx)
                if self.check_legal_action(batch_states[idx], action, extra_info):
                    actions.append(action)
                    actions_idx.append(action_idx)
                    break
        actions_idx = torch.as_tensor(actions_idx)
        actions_idx = F.one_hot(actions_idx, num_classes=62)
        target_q = actions_idx * q_hat
        target_q = torch.sum(target_q, dim=1)

        # compute target q
        target_q = batch_rewards + (1 - batch_dones) * self.gamma * target_q

        if self.flip == 1:
            self.q2.train()
            q_states = self.q2(batch_states)
            optim = self.optim2
        else:
            self.q1.train()
            q_states = self.q1(batch_states)
            optim = self.optim1

        batch_actions = F.one_hot(batch_actions, num_classes=62)
        predicted_q =batch_actions * q_states
        predicted_q = torch.sum(predicted_q, dim=1)

        # backward prop
        loss = F.huber_loss(input=predicted_q, target=target_q)
        loss.backward()
        # Update weights
        optim.step()
        optim.zero_grad()

    def train(self, max_episodes: int):
        episodes_rewards = []
        avg_losses = []

        for e in trange(max_episodes):
            ep_reward, avg_loss, eps = self.run_episode()

            episodes_rewards.append(ep_reward)
            avg_losses.append(avg_loss)

            if e % 50 == 0:
                print(f"Epsiode {e}.")
                print(f"Reward is {ep_reward}.")
                print(f"Loss is {avg_loss}.")
                print(f"Avg reward so far is {sum(episodes_rewards)/len(episodes_rewards)}.")
                print(f"Avg reward so far is {sum(avg_losses)/len(avg_losses)}.")
                torch.save(self.q1.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\macao_ddqn_q1_{e}.model")
                torch.save(self.q2.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\macao_ddqn_q2_{e}.model")

    def save_models(self):
        torch.save(self.q1.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\macao_ddqn_q1.model")
        torch.save(self.q2.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\macao_ddqn_q2.model")


if __name__ == "__main__":
    env = MacaoEnv()

    macao_agent = MacaoAgent(env=env, gamma=0.9, batch_size=64, lr=1e-3, pre_train_steps=100,
                             eps_scheduler=LinearScheduleEpsilon(start_eps=1, final_eps=0.05, pre_train_steps=100,
                                                                 final_eps_step=10 ** 4))
    macao_agent.train(max_episodes=10**4)
    macao_agent.save_models()