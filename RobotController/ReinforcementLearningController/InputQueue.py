from collections import deque
import numpy as np
from RobotController.RLConstants import INPUT_LAST_ACTIONS, ACTIONS_DEFINITION


class InputQueue:
    def __init__(self, capacity = INPUT_LAST_ACTIONS, action_size=ACTIONS_DEFINITION):
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity-1)
        self.action_size = action_size

    def push_state(self, state):
        self.states.append(state)
    def push_action(self, action):
        self.actions.append(action)

    def get_composed_state(self):
        # State is composed by the previous states concatenated with a concatenation of the previous actions hot-encoded
        return np.concatenate((np.array(self.states).flatten(),
                               np.eye(N=self.action_size, dtype=np.float32)[self.actions].flatten()))

    def __len__(self):
        return len(self.states)