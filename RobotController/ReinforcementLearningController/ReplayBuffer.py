from collections import deque
from numpy import array, float32
from random import sample
from RobotController.ReinforcementLearningController.RLConstants import DQN_REPLAY_BUFFER_CAPACITY, DQN_BATCH_SIZE


class ReplayBuffer(object):
    def __init__(self, capacity = DQN_REPLAY_BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size = DQN_BATCH_SIZE):
        state, action, reward, next_state = zip(*sample(self.buffer, batch_size))
        return array(state, dtype=float32), action, reward, array(next_state, dtype=float32)

    def state_sample(self, batch_size = DQN_BATCH_SIZE):
        state, _, _, _ = zip(*sample(self.buffer, k=batch_size))
        return array(state, dtype=float32)

    def __len__(self):
        return len(self.buffer)