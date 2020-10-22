from RobotController.ReinforcementLearningController.DQN import DQN
from RobotController.ReinforcementLearningController.World import World
from torch import no_grad
from RobotController.RLConstants import *
from os.path import join
from Constants import DEFAULT_PERSON_TO_FOLLOW, DECIMALS
from RobotController.RLConstants import PLAY_SESSION_TIME_IN_SECONDS
from numpy import mean
from time import time

class Validator:
    def __init__(self, input_size = len(STATES_ORDER), action_size=len(ACTIONS_DEFINITION), env = None,
                 charge_data_from = RL_CONTROLLER_DIR, model_dir = RL_CONTROLLER_PTH_FILE,
                 person_to_follow = DEFAULT_PERSON_TO_FOLLOW, session_time = PLAY_SESSION_TIME_IN_SECONDS):
        """
        Include the double Q network and is in charge of train and manage it
        :param input_size:
        :param action_size:
        :param buffer_size: int. Size of the replay states
        :param batch_size: int. Size of the Batch
        """
        # Instantiate both models
        self.current_model = DQN(input_size=input_size, num_actions=action_size)
        if charge_data_from is not None:
            self.current_model.load_weights(path=join(charge_data_from, model_dir))
        else:
            raise FileNotFoundError("Trying to play in validation mode but lacking a pth ControllerModel")

        self.input_size = input_size
        self.action_size = action_size
        self.env = env if env is not None else World(objective_person=person_to_follow)
        self.person_to_follow = person_to_follow
        self.charge_data_from = charge_data_from
        self.session_time = session_time

    def get_action(self, state):
        return self.current_model.act(state, epsilon=0.)

    def validate(self, verbose=True, show=False, steps_between_verbose = 10):
        """
        Train the network in the given environment for an amount of frames
        :param env:
        :param train_episodes:
        :return:
        """
        with no_grad():
            start_time = time()
            if verbose:
                print("Starting to game with {person}".format(person=self.person_to_follow))
            state, reward = self.env.step(IDLE)
            rewards = []
            while(time()-start_time < self.session_time):
                # Gets an action for the current state having in account the current epsilon
                action = self.get_action(state=state)
                if show:
                    self.env.render()
                # Execute the action, capturing the new state, the reward and if the game is ended or not
                state, reward = self.env.step(action)
                rewards.append(reward)
                if (len(rewards) % steps_between_verbose) == 0:
                    print("Remaining session time: {time} s\n"
                          "Mean reward: {reward}".format(time=round(self.session_time-(time()-start_time), ndigits=DECIMALS),
                                                         reward=round(mean(rewards), ndigits=DECIMALS)))



