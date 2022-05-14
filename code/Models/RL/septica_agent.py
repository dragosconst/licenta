from typing import List, Tuple, Dict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import trange

from Models.RL.Envs.septica import SepticaEnv
from Models.RL.rl_agent_utils import LinearScheduleEpsilon, ReplayBuffer

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
        player_hand, first_card, used_cards, value, is_first, is_challenging = x
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

        features = torch.stack((player_proj, first_proj, used_proj, value, is_first, is_challenging))

        h = self.hidden_layers(features)
        output = self.output_layer(h)
        return output

class SepticaAgent:
    def __init__(self, env: SepticaEnv, replay_buff_size=100, gamma=0.9, batch_size=512, lr=1e-3, steps_per_dqn=20,
                 pre_train_steps=1, eps_scheduler=LinearScheduleEpsilon(), max_steps_per_episode: int=200,
                 target_update_freq: int=10**3, tau: float=0.01):

        self.env = env
        self.num_actions = self.env.action_space.n
        self.target_update_freq = target_update_freq
        self.tau = tau

        self.replay_buff_size = replay_buff_size
        self.replay_buffer = ReplayBuffer(buffer_size=self.replay_buff_size)
        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode

        self.lr = lr
        self.q = SepticaModel("cpu")
        # gradient clipping
        for p in self.q.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        self.optim = Adam(self.q.parameters(), lr=self.lr)
        self.q_target = SepticaModel("cpu")
        self.batch_size = batch_size
        self.steps_per_dqn = steps_per_dqn
        self.pre_train_steps = pre_train_steps
        self.total_steps = 0
        self.q_updates = 0
        self.eps_scheduler = eps_scheduler

    def step(self, action: int, extra_info: str=None) -> Tuple[Tuple, ...]:
        state, reward, done = self.env.step(action=action, extra_info=extra_info)
        return state, reward, done

    def process_state(self, state):
        player_hand, first_card, used_cards, value, is_first, is_challenging = state
        new_player_hand = torch.zeros(54)
