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

NUM_ACTIONS = 74
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

        self.output_layer = nn.Linear(in_features=80, out_features=NUM_ACTIONS)

    def forward(self, x):
        # x is tuple of (hand, cards_pot, player_turns, suites, just_put, deck_np), where each element is already preprocessed for the net
        # the preprocessing consists of turning them to their one hot encoding and summing them
        # and turning them to tensors
        hand, card_pot, player_turns, suites, just_put, deck_no = zip(*x)
        hand = torch.stack(hand)
        card_pot = torch.stack(card_pot)
        player_turns = torch.stack(player_turns)
        suites = torch.stack(suites)
        just_put = torch.stack(just_put)
        deck_no = torch.stack(deck_no)

        hand = hand.to(self.device)
        card_pot = card_pot.to(self.device)
        player_turns = player_turns.to(self.device)
        suites = suites.to(self.device)
        just_put = just_put.to(self.device)
        deck_no = deck_no.to(self.device)

        hand_proj = self.hand_proj(hand)
        card_proj = self.pot_proj(card_pot)
        suites = self.suite_proj(suites)

        result = torch.cat((hand_proj, card_proj, player_turns, suites, just_put, deck_no), dim=1)

        h = self.hidden_layers(result)
        output = self.output_layer(h)
        return output


class MacaoAgent:
    def __init__(self, env: MacaoEnv, replay_buff_size=100, gamma=0.9, batch_size=512, lr=1e-3, steps_per_dqn=20,
                 pre_train_steps=1, eps_scheduler=LinearScheduleEpsilon(), max_steps_per_episode: int=200):
        self.env = env
        self.full_deck = build_deck()
        self.full_7s = ["7s", "7c", "7d", "7h"]
        self.full_suits = ["s", "c", "d", "h"]
        self.num_actions = self.env.action_space.n

        self.replay_buff_size = replay_buff_size
        self.replay_buffer = ReplayBuffer(buffer_size=self.replay_buff_size)
        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode

        self.lr = lr
        self.q1 = MacaoModel("cuda")
        self.q1.to("cuda")
        # gradient clipping
        for p in self.q1.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        self.optim1 = Adam(self.q1.parameters(), lr=self.lr)
        self.q2 = MacaoModel("cuda")
        self.q2.to("cuda")
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
        just_moved = 1 if just_moved else 0
        just_moved = torch.as_tensor([just_moved])


        return new_hand, new_last, torch.as_tensor([player_turns]), new_suits, just_moved, torch.as_tensor([deck])

    def check_legal_action(self, state, action, extra_info):
        hand, cards_pot, player_turns, suits, just_moved, deck = state
        new_cards_pot = [None] * 5
        for idx, card in enumerate(cards_pot):
            if card:
                new_cards_pot[int(card.item()) - 1] = self.full_deck[idx]
        new_hand = set()
        for idx, card in enumerate(hand):
            if card:
                new_hand.add(self.full_deck[idx])

        if action == 0:
            if player_turns > 0:
                return False
            if extra_info not in new_hand:
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
            if just_moved and new_cards_pot[-1][0] == "A":
                return extra_info[0] == "A"

            # finally, we are left with the case of trying to put a regular card over another regular card
            return new_cards_pot[-1][0] == extra_info[0] or same_suite(suits, extra_info)
        elif action == 1:
            if player_turns > 0:
                return False
            if len([card for card in cards_pot if card != 0]) <= 1 and deck == 0:
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
            if just_moved and new_cards_pot[-1][0] == "A":
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
            if extra_info[:2] not in new_hand:
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
            if just_moved and new_cards_pot[-1][0] == "A":
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
            if just_moved and new_cards_pot[-1][0] == "A":
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
        - 57-72: change suite
        - 73: don't do anything, used when waiting turns
        """
        if idx <= 53:
            return 0, self.full_deck[idx]
        elif idx == 54:
            return 1, None
        elif idx == 55:
            return 2, None
        elif idx == 56:
            return 3, None
        elif idx <= 72:
            return 4, self.full_7s[(idx - 57)//4] + " " + self.full_suits[(idx - 1) % 4]
        elif idx == 73:
            return 5, None

    def action_to_idx(self, action, extra_info=None):
        """
        Reverse of earlier method.
        """
        if action == 0:
            idx = self.full_deck.index(extra_info)
            return idx
        elif action == 1:
            return 54
        elif action == 2:
            return 55
        elif action == 3:
            return 56
        elif action == 4:
            return 57 + self.full_7s.index(extra_info[:2]) * 4 + self.full_suits.index(extra_info[-1])
        elif action == 5:
            return 73

    def get_legal_actions(self, state) -> List[Tuple]:
        """
        Get all legal actions from a given state.
        """
        hand, *_ = state
        new_hand = set()
        for idx, card in enumerate(hand):
            if card:
                new_hand.add(self.full_deck[idx])

        actions = []
        for action in [0, 1, 2, 3, 4, 5]:
            if action == 0:
                for card in new_hand:
                    if self.check_legal_action(state, action, card):
                        actions.append((action, card))
            elif action <= 3:
                if self.check_legal_action(state, action, None):
                    actions.append((action, None))
            elif action == 4:
                for card in new_hand:
                    if card[0] == "7":
                        for suit in self.full_suits:
                            if self.check_legal_action(state, action, card + " " + suit):
                                actions.append((action, card + " " + suit))
            elif action == 5:
                if self.check_legal_action(state, action, None):
                    actions.append((action, None))
        return actions

    @torch.inference_mode()
    def get_action(self, states, eps):
        p = random.random()

        if self.total_steps <= self.pre_train_steps or p < eps:
            actions = []
            for state in states:
                possible_actions = self.get_legal_actions(state)
                action, extra_info = random.sample(possible_actions, 1)[0]
                actions.append((action, extra_info))
        else:
            if self.flip == 1:
                q_hat = self.q1(states)
            else:
                q_hat = self.q2(states)

            q_hat_sorted = torch.argsort(q_hat, dim=1)
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
        for ep_step in range(self.max_steps_per_episode):
            epsilon = self.eps_scheduler.get_value(step=self.total_steps)

            batch = [current_state]
            self.flip = random.choice([1, 2])
            actions = self.get_action(batch, epsilon)
            current_action, current_extra_info = actions[0]

            new_state, reward, done = self.step(current_action, current_extra_info)
            new_state_proc = self.process_state(new_state)
            ep_reward += reward

            self.replay_buffer.add(state=current_state, action=self.action_to_idx(current_action, current_extra_info), reward=reward, next_state=new_state_proc, done=done)

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
        actions_idx = []
        for idx, action_idxs in enumerate(q_hat_sorted):
            for action_idx in reversed(action_idxs):
                action, extra_info = self.idx_to_action(action_idx)
                if self.check_legal_action(batch_states[idx], action, extra_info):
                    actions_idx.append(action_idx)
                    break
        actions_idx = torch.as_tensor(actions_idx)
        actions_idx = F.one_hot(actions_idx, num_classes=NUM_ACTIONS)
        target_q = actions_idx * q_hat.to("cpu")
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

        batch_actions = F.one_hot(batch_actions, num_classes=NUM_ACTIONS)
        predicted_q = batch_actions * q_states.to("cpu")
        predicted_q = torch.sum(predicted_q, dim=1)

        # backward prop
        loss = F.huber_loss(input=predicted_q, target=target_q)
        loss.backward()
        # Update weights
        optim.step()
        optim.zero_grad()
        return loss

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
                print(f"Avg loss so far is {sum(avg_losses)/len(avg_losses)}.")
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