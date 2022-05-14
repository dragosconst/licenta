import collections
import random


# it's literally just a deque with a max size and specialized sampling function
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

        self.replay_mem = collections.deque(maxlen=buffer_size)

    # Add a new experience to the buffer
    def add(self, state, action, reward, next_state, done):
        self.replay_mem.append((state, action, reward, next_state, done))

    # Get a sample batch from the memory
    def sample(self, batch_size):
        assert batch_size <= len(self.replay_mem)
        return random.sample(self.replay_mem, batch_size)

    def __len__(self):
        return len(self.replay_mem)


# linear scheduler for eps
class LinearScheduleEpsilon():
    def __init__(self, start_eps=1.0, final_eps=0.1,
                 pre_train_steps=10, final_eps_step=10000):
        self.start_eps = start_eps
        self.final_eps = final_eps
        self.pre_train_steps = pre_train_steps
        self.final_eps_step = final_eps_step
        self.decay_per_step = (self.start_eps - self.final_eps) / (self.final_eps_step - self.pre_train_steps)

    def get_value(self, step):
        if step <= self.pre_train_steps:
            return 1.0  # full exploration in the beginning
        else:
            eps_value = (1.0 - self.decay_per_step * (step - self.pre_train_steps))
            eps_value = max(self.final_eps, eps_value)
            return eps_value
