from collections import deque
import numpy as np
from RobotController.RLConstants import INPUT_LAST_ACTIONS, ACTIONS_DEFINITION, ROTATION_ADVANCE_BY_ACTION


class InputQueue:
    def __init__(self, capacity = INPUT_LAST_ACTIONS, action_size=ACTIONS_DEFINITION):
        """
        Queue used for the DQN for including information about past states.
        :param capacity: Int. Amount of states to include in each DQN input.
        :param action_size: Int. Amount of actions that can be decided by the DQN.
        """
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity-1)
        self.action_size = action_size

    def push_state(self, state):
        """
        Pushes a state into the Queue
        :param state: List of float. Current state of the environment
        """
        self.states.append(state)

    def push_action(self, action):
        """
        Pushes an action into the Queue
        :param state: Float. Last action taken.
        """
        self.actions.append(action)


    def get_composed_state(self):
        """
        Returns the list of states composed as a 2-dimensional numpy.
        :return: 2D Numpy. List of states. The action is translated to 2 values: (Rotation, Advance).
        """
        return np.stack([np.concatenate([state, ROTATION_ADVANCE_BY_ACTION[action]]) for state, action in zip(self.states, list(self.actions)+[-1])])

    def __len__(self):
        return len(self.states)