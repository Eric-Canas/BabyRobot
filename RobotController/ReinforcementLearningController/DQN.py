from random import random, randrange
from torch import load, save, FloatTensor, no_grad, device, zeros
from torch.nn import Sequential, Linear, ReLU, Module, Dropout, BatchNorm1d, LeakyReLU, LSTM
from torch.cuda import is_available
from os.path import join, dirname, isdir, isfile
from os import makedirs
from RobotController.RLConstants import RL_CONTROLLER_DIR, RL_CONTROLLER_PTH_FILE
from warnings import warn

dev = device('cuda') if is_available() else device('cpu')

print("Working on: {dev}".format(dev=dev))

class DQN(Module):
    def __init__(self, input_size, num_actions, lstm_hidden_size = 64, lstm_layers = 1):
        super(DQN, self).__init__()
        # Linear Based

        self.hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm = LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.lstm_layers, batch_first=True)

        self.network = Sequential(Linear(self.hidden_size, 128),
                                  LeakyReLU(inplace=True),
                                  BatchNorm1d(num_features=128),
                                  Dropout(p=0.5, inplace=False),
                                  Linear(128, 64),
                                  LeakyReLU(inplace=True),
                                  BatchNorm1d(num_features=64),
                                  Dropout(p=0.5, inplace=False),
                                  Linear(64, num_actions))

        self.num_actions = num_actions
        self.input_size = input_size
        self.to(device=dev)

    def forward(self, x):
        h0 = zeros(self.lstm_layers, x.size(0), self.hidden_size).to(device=dev)
        c0 = zeros(self.lstm_layers, x.size(0), self.hidden_size).to(device=dev)
        x, _ = self.lstm(x, (h0, c0))
        # Take only the last state of the output
        return self.network(x[:, -1, ...])

    def act(self, state, epsilon=0.):
        if random() > epsilon:
            self.eval()
            with no_grad():
                state = FloatTensor(state).to(device=dev)
                q_value = self.forward(state[None,:])[0]
                action = q_value.argmax().item()
            self.train()
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