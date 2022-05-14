from typing import List, Tuple, Dict
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import trange

from Models.RL.Envs.septica import SepticaEnv
from Models.RL.rl_agent_utils import LinearScheduleEpsilon, ReplayBuffer
from Models.RL.Envs.septica_utils import build_deck

NUM_ACTIONS = 33
class SepticaModel(nn.Module):
    def __init__(self, device: str):
        super(SepticaModel, self).__init__()
        self.device = device

        self.hand_proj = nn.Linear(in_features=54, out_features=25)
        self.first_proj = nn.Linear(in_features=54, out_features=20)
        self.used_proj = nn.Linear(in_features=54, out_features=30)

        self.hidden_layers = nn.Sequential(
            nn.Linear(in_features=78, out_features=220),
            nn.ReLU(),
            nn.Linear(in_features=220, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=100),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(in_features=100, out_features=NUM_ACTIONS)

    def forward(self, x):
        """
        States should be sent-in already processed.
        """
        player_hand, first_card, used_cards, value, is_first, is_challenging = zip(*x)
        player_hand = torch.stack(player_hand)
        first_card = torch.stack(first_card)
        used_cards = torch.stack(used_cards)
        value = torch.stack(value)
        is_first = torch.stack(is_first)
        is_challenging = torch.stack(is_challenging)

        player_hand = player_hand.to(self.device)
        first_card = first_card.to(self.device)
        used_cards = used_cards.to(self.device)
        value = value.to(self.device)
        is_first = is_first.to(self.device)
        is_challenging = is_challenging.to(self.device)

        player_proj = self.hand_proj(player_hand)
        first_proj = self.first_proj(first_card)
        used_proj = self.used_proj(used_cards)

        features = torch.cat((player_proj, first_proj, used_proj, value, is_first, is_challenging), dim=1)

        h = self.hidden_layers(features)
        output = self.output_layer(h)
        return output

class SepticaAgent:
    def __init__(self, env: SepticaEnv, replay_buff_size=100, gamma=0.9, batch_size=512, lr=1e-3, steps_per_dqn=20,
                 pre_train_steps=1, eps_scheduler=LinearScheduleEpsilon(), max_steps_per_episode: int=200,
                 target_update_freq: int=10**3, tau: float=0.01):

        self.env = env
        self.num_actions = self.env.action_space.n
        self.full_deck = build_deck()
        self.target_update_freq = target_update_freq
        self.tau = tau

        self.replay_buff_size = replay_buff_size
        self.replay_buffer = ReplayBuffer(buffer_size=self.replay_buff_size)
        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode

        self.lr = lr
        self.q = SepticaModel("cpu")
        self.q.to(self.q.device)
        # gradient clipping
        for p in self.q.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        self.optim = Adam(self.q.parameters(), lr=self.lr)
        self.q_target = SepticaModel("cpu")
        self.q_target.to(self.q_target.device)
        self.batch_size = batch_size
        self.steps_per_dqn = steps_per_dqn
        self.pre_train_steps = pre_train_steps
        self.total_steps = 0
        self.q_updates = 0
        self.eps_scheduler = eps_scheduler

    def step(self, action: int, extra_info: str=None) -> Tuple[Tuple, ...]:
        state, reward, done = self.env.step(action=action, extra_info=extra_info)
        return state, reward, done

    def process_state(self, state) -> Tuple:
        player_hand, first_card, used_cards, value, is_first, is_challenging = state
        new_player_hand = torch.zeros(54)
        for card in player_hand:
            new_player_hand[self.full_deck.index(card)] = 1
        new_first = torch.zeros(54)
        if first_card is not None:
            new_first[self.full_deck.index(first_card)] = 1
        new_used_cards = torch.zeros(54)
        for card in used_cards:
            new_used_cards[self.full_deck.index(card)] = 1
        is_first = 1 if is_first else 0
        is_challenging = 1 if is_challenging else 0

        return new_player_hand, new_first, new_used_cards, torch.as_tensor([value]), torch.as_tensor([is_first]),\
               torch.as_tensor([is_challenging])


    def check_legal_action(self, state, action, extra_info=None):
        player_hand_one_hot, first_card_one_hot, used_cards, value, is_first, is_challenging = state
        player_hand = []
        for idx, card in enumerate(player_hand_one_hot):
            if card:
                player_hand.append(self.full_deck[idx])
        first_card = None
        for idx, card in enumerate(first_card_one_hot):
            if card:
                first_card = self.full_deck[idx]
                break
        is_first = is_first[0].item()
        is_challenging = is_challenging[0].item()

        if action == 0:
            assert extra_info is not None
            card = extra_info
            if card not in player_hand:
                return False
            if is_first and first_card is None:
                return True
            if is_first and is_challenging:
                return card[0] == "7" or card[0] == first_card[0]
            elif is_first and not is_challenging and first_card is not None:
                return False
            return True
        elif action == 1:
            return is_first and first_card is not None

    def get_legal_actions(self, state) -> List:
        hand_one_hot, *_ = state
        hand = []
        for idx, card in enumerate(hand_one_hot):
            if card:
                hand.append(self.full_deck[idx])

        actions = []
        for action in [0, 1]:
            if action == 0:
                for card in hand:
                    if self.check_legal_action(state, action, card):
                        actions.append((action, card))
            elif action == 1:
                if self.check_legal_action(state, action):
                    actions.append((action, None))
        return actions

    def idx_to_action(self, idx) -> Tuple:
        """
        Convert action index to (action, extra_info) tuple.
        """
        if idx < len(self.full_deck):
            return 0, self.full_deck[idx]
        return 1, None

    def action_to_idx(self, action, extra_info=None):
        """
        Reverse of precedent function.
        """
        if action == 0:
            return self.full_deck.index(extra_info)
        return len(self.full_deck)

    def get_action(self, states, eps):
        p = random.random()

        if self.total_steps <= self.pre_train_steps or p < eps:
            actions = []
            for state in states:
                possible_actions = self.get_legal_actions(state)
                action, extra_info = random.sample(possible_actions, 1)[0]
                actions.append((action, extra_info))
        else:
            q_hat = self.q(states)
            q_hat_sorted = torch.argsort(q_hat, dim=1)
            actions = [None] * len(q_hat_sorted)
            for idx, q_actions_indexes in enumerate(q_hat_sorted):
                for aidx in reversed(q_actions_indexes):
                    action, extra_info = self.idx_to_action(aidx)
                    if self.check_legal_action(states[idx], action, extra_info):
                        actions[idx] = (action, extra_info)
                        break
        return actions

    def run_episode(self):
        """
        Run a single training episode
        """
        init = self.env.reset()

        ep_reward = 0
        current_state = self.process_state(init)
        total_loss = 0
        num_loss_comp = 0
        epsilon = 1.0
        for ep_step in range(self.max_steps_per_episode):
            epsilon = self.eps_scheduler.get_value(step=self.total_steps)

            batch = [current_state]
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
        batch_nexts = [x[3] for x in train_batch]
        batch_dones = torch.as_tensor([x[4] for x in train_batch])

        self.q.eval()
        self.q_target.eval()
        self.optim.zero_grad()
        with torch.inference_mode():
            q_next = self.q(batch_nexts)
            q_next_target = self.q_target(batch_nexts)
        # select best legal action according to regular Q
        q_next_sorted = torch.argsort(q_next, dim=1)
        actions_as_idx_batch = [None] * len(q_next_sorted)
        # execution bottleneck
        for idx, actions_as_idx_state in enumerate(q_next_sorted):
            for action_idx in reversed(actions_as_idx_state):
                action, extra_info = self.idx_to_action(action_idx)
                if self.check_legal_action(batch_states[idx], action, extra_info):
                    actions_as_idx_batch[idx] = action_idx
                    break
        actions_as_idx_batch = torch.as_tensor(actions_as_idx_batch)
        actions_as_idx_batch = F.one_hot(actions_as_idx_batch, num_classes=NUM_ACTIONS)
        target_q = actions_as_idx_batch * q_next_target.cpu()  # use values estimated by q_target
        target_q = torch.sum(target_q, dim=1)
        # compute target q
        target_q = batch_rewards + (1 - batch_dones) * self.gamma * target_q

        self.q.train()
        q_states = self.q(batch_states)
        batch_actions = F.one_hot(batch_actions, num_classes=NUM_ACTIONS)
        predicted_q = batch_actions * q_states.cpu()
        predicted_q = torch.sum(predicted_q, dim=1)

        # backward prop
        loss = F.huber_loss(input=predicted_q, target=target_q)
        loss.backward()
        # Update weights
        self.optim.step()

        # q_target weights update with polyak averaging -> way better than weights update at a specific time step
        for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(self.tau * param - (1 - self.tau) * target_param)
        return loss

    def train(self, max_episodes: int):
        episodes_rewards = []
        avg_losses = []
        print_freq = 20
        save_freq = 50
        running_mean_reward = deque(maxlen=print_freq)

        for e in trange(max_episodes):
            ep_reward, avg_loss, eps = self.run_episode()

            episodes_rewards.append(ep_reward)
            running_mean_reward.append(ep_reward)
            avg_losses.append(avg_loss)

            if e % print_freq == 0:
                print(f"Epsiode {e}.")
                print(f"Reward is {ep_reward}.")
                print(f"Loss is {avg_loss}.")
                print(f"Eps is {self.eps_scheduler.get_value(step=self.total_steps)}")
                print(f"Avg reward so far is {sum(episodes_rewards)/len(episodes_rewards)}.")
                print(f"Reward of last {print_freq} eps is {sum(running_mean_reward)/len(running_mean_reward)}")
                print(f"Avg loss so far is {sum(avg_losses)/len(avg_losses)}.")
            if e % save_freq == 0:
                torch.save(self.q.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_q_{e}_doubleq.model")
                torch.save(self.q_target.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_qtarget_{e}_doubleq.model")

    def save_models(self):
        torch.save(self.q.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_q_doubleq.model")
        torch.save(self.q_target.state_dict(), f"D:\\facultate stuff\\licenta\\data\\rl_models\\septica_ddqn_qtarget_doubleq.model")


if __name__ == "__main__":
    env = SepticaEnv()

    macao_agent = SepticaAgent(env=env, gamma=1, batch_size=64, replay_buff_size=512, lr=1e-3, pre_train_steps=300,
                             eps_scheduler=LinearScheduleEpsilon(start_eps=1, final_eps=0.05, pre_train_steps=300,
                                                                 final_eps_step=5*10 ** 4))
    macao_agent.train(max_episodes=10**4)
    macao_agent.save_models()