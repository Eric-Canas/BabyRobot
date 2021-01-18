from collections import deque
from numpy import array, float32
from random import sample
from RobotController.RLConstants import DQN_REPLAY_BUFFER_CAPACITY, DQN_BATCH_SIZE, USUAL_CASE_USE_PROB
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity = DQN_REPLAY_BUFFER_CAPACITY, usual_case_probability = USUAL_CASE_USE_PROB):
        """
        Replay buffer for the DQN. In order to improve the training of the DQN, if it is at more than at 75% of its
        capacity, it only saves the states that had a reward higher than the average, lower than average-2*std, or
        any state with a probability of usual_case_probability.
        :param capacity: Int. Capacity of the Queue.
        :param usual_case_probability: Float. Probability of including an Input in the average.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.usual_case_probability = usual_case_probability
        if self.usual_case_probability is not None:
            self.rewards = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """
        Push an Input into the queue. If it is at more than at 75% of its capacity, it only saves the input if it had
        a reward higher than the average, lower than average-2*std, or  any state with a probability
        of usual_case_probability
        :param state: List of Float. Current state of the environment
        :param action: Int. Action taken by the DQN
        :param reward: Float. Reward obtained y executing the action.
        :param next_state: List of Float. Next state. Resulting of executing the given action in to the state.
        """
        is_useful = True
        if len(self.buffer) > self.capacity*0.75  and self.usual_case_probability is not None:
            mean, std = np.mean(self.rewards), np.std(self.rewards)
            is_useful = (reward > mean) or (reward < (mean-std*2)) or (np.random.uniform() <= self.usual_case_probability)

        if is_useful:
            self.buffer.append((state, action, reward, next_state))
            if self.usual_case_probability is not None:
                self.rewards.append(reward)

    def sample(self, batch_size = DQN_BATCH_SIZE):
        """
        Get a batch with batch_size inputs
        :param batch_size: Int. Number of  inputs to return as a batch.
        :return: 3D-Numpy. Input batch for the DQN (BATCH x SEQUENCE x INPUT)
        """
        state, action, reward, next_state = zip(*sample(self.buffer, batch_size))
        return array(state, dtype=float32), action, reward, array(next_state, dtype=float32)

    def state_sample(self, batch_size = DQN_BATCH_SIZE):
        """"
        Get a batch with batch_size states
        :param batch_size: Int. Number of  states to return as a batch.
        :return: 3D-Numpy. States batch (BATCH x SEQUENCE x STATE)
        """
        state, _, _, _ = zip(*sample(self.buffer, k=batch_size))
        return array(state, dtype=float32)

    def __len__(self):
        return len(self.buffer)