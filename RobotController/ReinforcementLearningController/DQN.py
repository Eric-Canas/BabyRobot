from random import random, randrange
from torch import load, save, FloatTensor, no_grad, device
from torch.nn import Sequential, Linear, ReLU, Module, Dropout, BatchNorm1d, LeakyReLU
from torch.cuda import is_available
from os.path import join, dirname, isdir, isfile
from os import makedirs
from RobotController.RLConstants import RL_CONTROLLER_DIR, RL_CONTROLLER_PTH_FILE
from warnings import warn

dev = device('cuda') if is_available() else device('cpu')

print("Working on: {dev}".format(dev=dev))

class DQN(Module):
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.network = Sequential(Linear(input_size, 256),
                                  LeakyReLU(inplace=True),
                                  BatchNorm1d(256),
                                  Dropout(p=0.5, inplace=False),
                                  Linear(256, 64),
                                  LeakyReLU(inplace=True),
                                  BatchNorm1d(64),
                                  Dropout(p=0.5, inplace=False),
                                  Linear(64, num_actions))
        self.num_actions = num_actions
        self.input_size = input_size
        self.to(device=dev)

    def forward(self, x):
        return self.network(x)

    def act(self, state, epsilon=0.):
        if random() > epsilon:
            with no_grad():
                state = FloatTensor(state).to(device=dev)
                q_value = self.forward(state)
                action = q_value.argmax().item()
        else:
            action = randrange(self.num_actions)
        return action

    def save(self, path=join(RL_CONTROLLER_DIR, RL_CONTROLLER_PTH_FILE)):
        if not isdir(dirname(path)):
            makedirs(dirname(path))
        self.cpu()
        save(self.state_dict(), path)
        self.to(device=dev)

    def load_weights(self, path=join(RL_CONTROLLER_DIR, RL_CONTROLLER_PTH_FILE)):
        if isfile(path):
            self.cpu()
            self.load_state_dict(load(path))
            self.to(device=dev)
        else:
            warn('Trying to load a non-existing model. Load Skipped.')
        return self