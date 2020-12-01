from collections import deque
from numpy import array, float32
from random import sample
from RobotController.RLConstants import DQN_REPLAY_BUFFER_CAPACITY, DQN_BATCH_SIZE
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity = DQN_REPLAY_BUFFER_CAPACITY, usual_case_probability = 0.01):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.usual_case_probability = usual_case_probability
        if self.usual_case_probability is not None:
            self.rewards = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        is_useful = True
        if len(self.buffer) > self.capacity*0.75  and self.usual_case_probability is not None:
            mean, std = np.mean(self.rewards), np.std(self.rewards)
            is_useful = (reward > mean) or (reward < (mean-std*2)) or (np.random.uniform() <= self.usual_case_probability)

        if is_useful:
            self.buffer.append((state, action, reward, next_state))
            if self.usual_case_probability is not None:
                self.rewards.append(reward)

    def sample(self, batch_size = DQN_BATCH_SIZE):
        state, action, reward, next_state = zip(*sample(self.buffer, batch_size))
        return array(state, dtype=float32), action, reward, array(next_state, dtype=float32)

    def state_sample(self, batch_size = DQN_BATCH_SIZE):
        state, _, _, _ = zip(*sample(self.buffer, k=batch_size))
        return array(state, dtype=float32)

    def __len__(self):
        return len(self.buffer)