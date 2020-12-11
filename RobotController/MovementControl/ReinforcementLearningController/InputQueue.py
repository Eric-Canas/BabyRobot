from collections import deque
import numpy as np
from RobotController.RLConstants import INPUT_LAST_ACTIONS, ACTIONS_DEFINITION, ROTATION_ADVANCE_BY_ACTION


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
        return np.stack([np.concatenate([state, ROTATION_ADVANCE_BY_ACTION[action]]) for state, action in zip(self.states, list(self.actions)+[-1])])

    def __len__(self):
        return len(self.states)