from math import sqrt, pi
from RobotController.ReinforcementLearningController.DQN import DQN, dev
from RobotController.ReinforcementLearningController.World import World
from pickle import dump, load
from RobotController.ReinforcementLearningController.ReplayBuffer import ReplayBuffer
from torch import LongTensor, FloatTensor
from torch.nn.functional import smooth_l1_loss
from torch.optim import Adam, RMSprop
from RobotController.RLConstants import *
from warnings import warn
from os.path import join, isfile, isdir
from os import makedirs, listdir
from Constants import DECIMALS
from RobotController.RLConstants import INPUT_LAST_ACTIONS, PLAY_SESSION_TIME_IN_SECONDS, ACTIONS_TELEOPERATED_KEYPAD_DEFINITON, ROTATION_ADVANCE_BY_ACTION
import numpy as np
from matplotlib import pyplot as plt
from time import time
from RobotController.ReinforcementLearningController.InputQueue import InputQueue
from RobotController.ClientServer.ClientPipeline import ClientPipeline
from collections import deque
from random import random, randrange
from Utilities.GetKey import get_key
import torch
PLOT_EVERY = 100


class Trainer:
    def __init__(self, input_size = len(STATES_ORDER), action_size=len(ACTIONS_DEFINITION), gamma=GAMMA, buffer_size=DQN_REPLAY_BUFFER_CAPACITY,
                 batch_size=DQN_BATCH_SIZE, loss = smooth_l1_loss, env = None, clip_weights = True,
                 episodes_between_saving=EPISODES_BETWEEN_SAVING, charge_data_from = RL_CONTROLLER_DIR, save_data_at = RL_CONTROLLER_DIR,
                 model_dir = RL_CONTROLLER_PTH_FILE, session_time = PLAY_SESSION_TIME_IN_SECONDS, input_last_actions = INPUT_LAST_ACTIONS,
                 promote_improvement_in_reward=False, DQN_lr = DQN_LEARNING_RATE, send_security_copy=False, tele_operate_exploration = False,
                 verbose = True):
        """
        Include the double Q network and is in charge of train and manage it
        :param input_size:
        :param action_size:
        :param buffer_size: int. Size of the replay states
        :param batch_size: int. Size of the Batch
        """
        self.input_size = input_size+len(ROTATION_ADVANCE_BY_ACTION[-1])#(input_size*input_last_actions)+(action_size*(input_last_actions-1))
        self.input_buffer = InputQueue(capacity=input_last_actions, action_size = action_size)
        self.input_last_actions = input_last_actions
        self.buffer_size = buffer_size
        # Instantiate both models
        self.current_model = DQN(input_size=self.input_size, num_actions=action_size)
        if charge_data_from is not None:
            self.current_model.load_weights(path=join(charge_data_from, model_dir))

        self.target_model = DQN(input_size=self.input_size, num_actions=action_size)

        # Initialize the Adam optimizer and the replay states
        self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.current_model.parameters()),lr=DQN_lr)

        # Make both networks start with the same weights
        self.update_target()

        # Save the rest of parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.action_size = action_size
        self.env = env if env is not None else World(objective_person=None)
        self.executing_on_server = type(self.env.recognition_pipeline) is ClientPipeline
        if self.executing_on_server:
            self.socket = self.env.recognition_pipeline.socket
        self.loss = loss
        self.clip_weights = clip_weights
        self.episodes_between_saving = episodes_between_saving
        self.save_data_at = save_data_at
        self.charge_data_from = charge_data_from
        self.verbose = verbose
        self.step_time = deque(maxlen=EPISODES_BETWEEN_SAVING*3)
        self.buffer_influence = torch.exp((torch.arange(float(input_last_actions))/float(input_last_actions)).float())
        self.buffer_influence = self.buffer_influence / torch.sum(self.buffer_influence)
        if self.charge_data_from is None:
            self.losses, self.all_rewards = [], []
        else:
            self.losses, self.all_rewards, self.replay_buffer = self.charge_previous_losses_and_rewards(charge_from=self.charge_data_from)
        self.session_time = session_time
        # By default, this experiment is disabled. However, if it is determined that the robot should only
        # lose the person when it (otherwise) will collide with a piece of furniture, it could be activated.
        # (since this danger has higher influence in the reward than the distance to the person).
        if promote_improvement_in_reward:
            self.promote_improvement = lambda reward, last_reward: reward + ((reward-last_reward)*IMPROVEMENT_BONUS if reward > last_reward else 0)
        else:
            self.promote_improvement = lambda reward, last_reward: reward
        self.send_security_copy = send_security_copy
        self.tele_operate_exploration = tele_operate_exploration

    def get_action(self, state, epsilon = 0.):
        if random() > epsilon:
            action = self.current_model.act(state, epsilon=epsilon+EPSILON_FINAL)
        else:
            if self.tele_operate_exploration:
                action = self.request_for_action()
            else:
                action = randrange(self.action_size)
        return action

    def request_for_action(self, timeout=REQUEST_FOR_ACTION_TIMEOUT):
        action = get_key(timeout=timeout)
        if self.verbose: print("Teleoperated action introduced: {act}".format(act=action))
        if action == INVALID:
            action = randrange(self.action_size)
        else:
            action = ACTIONS_TELEOPERATED_KEYPAD_DEFINITON[action]

        return action
    def update_target(self):
        """
        Updates the target model with the weights of the current model
        """
        self.target_model.load_state_dict(self.current_model.state_dict())

    def compute_td_loss(self, samples):
        """
        Compute the loss of batch size samples of the states, and train the current model network with that loss
        :param samples: tuple of samples. Samples must have the format (state, action, reward, next_state)
        :return:
        float. Loss computed at this learning step
        """
        # Take N playing samples
        state, action, reward, next_state = samples

        # Transform them into torch variables, for being used on GPU during the training
        state = FloatTensor(np.float32(state)).to(device=dev)
        next_state = FloatTensor(np.float32(next_state)).to(device=dev)
        action = LongTensor(action).to(device=dev)
        reward = FloatTensor(reward).to(device=dev)

        # Predict the q value of this state and all the q values of the following step
        q_value = self.current_model(state).gather(dim=1, index=action[...,None]).flatten()
        next_q_values = self.current_model(next_state)
        predicted_actions = next_q_values.argmax(dim=1)[...,None]
        # Get the q values of the following step following the static policy of the target model
        next_q_state_values = self.target_model(next_state)
        # For all the q_values of the next state get the one of the action which would be selected by the static policy
        next_q_value = next_q_state_values.gather(dim=1, index=predicted_actions).flatten()
        # Calculate the expected q value as the immediate reward plus gamma by the expected reward at t+1 (if not ended)
        expected_q_value = reward + (self.gamma * next_q_value)

        # Calculate the Huber Loss
        loss = self.loss(q_value, expected_q_value)

        # Backpropagates the loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_weights:
            # Clip weights as the original DQN paper did (For avoiding to explode)
            for param in self.current_model.parameters():
                param.grad.data.clamp_(-1, 1)
        # Learn
        self.optimizer.step()

        # Return the loss of this step
        return loss



    def epsilon_by_step(self, step, epsilon_start = EPSILON_START, epsilon_final = EPSILON_FINAL, epsilon_decay = EPSILON_DECAY):
        """
        Gets the epsilon of the current frame for the given parameters
        :param step: int. Index of the frame
        :param epsilon_start: float. Epsilon at frame 1
        :param epsilon_final: float. Minimum epsilon for maintaining exploration
        :param epsilon_decay: int. Manages how fast the epsilon decays
        :return:
        Epsilon for the frame frame_idx
        """
        # epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * step / epsilon_decay) # Very hard decay
        return max(epsilon_start*epsilon_decay**step, epsilon_final)

    def train(self, train_episodes = TRAIN_STEPS, steps_per_episode = STEPS_PER_EPISODE,
              DQN_update_ratio = DQN_UPDATE_RATIO, show=False, episodes_between_saving = None):
        """
        Train the network in the given environment for an amount of frames
        :param env:
        :param train_episodes:
        :return:
        """

        episodes_between_saving = self.episodes_between_saving if episodes_between_saving is None else episodes_between_saving
        # Save the losses of the network and the rewards of each episode
        reward = 0
        for i in range(self.input_last_actions):
            self.input_buffer.push_action(action=IDLE)
            partial_state, reward = self.env.step(IDLE)
            self.input_buffer.push_state(state=partial_state)

        if self.verbose:
            if len(self.replay_buffer) < self.batch_size:
                print("Performing random movements for filling the batch")
            else:
                print("Charged a previous replay buffer with {n} actions".format(n=len(self.replay_buffer)))
        composed_state = self.input_buffer.get_composed_state()
        last_reward = reward
        while len(self.replay_buffer) < self.batch_size:
            action = self.get_action(state=composed_state, epsilon=1.)
            partial_next_state, reward = self.env.step(action)
            self.input_buffer.push_action(action=action)
            self.input_buffer.push_state(state=partial_next_state)
            composed_next_state = self.input_buffer.get_composed_state()
            improved_reward = self.promote_improvement(reward=reward, last_reward=last_reward)
            self.replay_buffer.push(composed_state, action, improved_reward, composed_next_state)
            composed_state = composed_next_state
            last_reward = reward

        if self.verbose:
            print("Starting the train!")
        start_time = time()

        for episode in range(1, train_episodes+1):
            episode_reward = 0
            actions_taken = []
            current_epsilon = self.epsilon_by_step(step=episode)
            episode_losses = []
            for step in range(1, steps_per_episode+1):

                # Gets an action for the current state having in account the current epsilon
                action = self.get_action(state=composed_state, epsilon=current_epsilon)
                actions_taken.append(action)
                if show:
                    self.env.render()
                # Execute the action, capturing the new state, the reward and if the game is ended or not
                if self.verbose: start_step_time = time()
                partial_next_state, reward = self.env.step(action)
                if self.verbose: self.step_time.append(time()-start_step_time)

                self.input_buffer.push_action(action=action)
                self.input_buffer.push_state(state=partial_next_state)
                composed_next_state = self.input_buffer.get_composed_state()
                # Save the action at the replay states
                improved_reward = self.promote_improvement(reward=reward, last_reward=last_reward)
                improved_reward += self.additional_maintaing_score(composed_next_state)
                #if improved_reward >= MAX_REWARD_BY_PARAM*WALL_DISTANCE_INFLUENCE and random()>SAVE_NON_REWARDED_STATE_PROB:
                self.replay_buffer.push(composed_state, action, improved_reward, composed_next_state)
                # Update the current state and the actual episode reward
                composed_state = composed_next_state
                episode_reward += reward
                last_reward = reward
                # If there are enough actions in the states for learning, start to learn a policy
                if step % ACTIONS_PER_TRAIN_STEP == 0:
                    # Train
                    loss = self.compute_td_loss(self.replay_buffer.sample(self.batch_size))
                    # Save the loss
                    episode_losses.append(loss.item())

                if step % DQN_update_ratio == 0:
                    self.update_target()

            self.all_rewards.append(episode_reward)
            self.losses.append(np.mean(episode_losses))
            # If a game is finished save the results of that game and restart the game
            if self.verbose:
                print("-"*50+'\n'
                      "Episode Reward: {epReward}\n"
                      "Std of actions: {std}\n"
                      "Epsilon: {epsilon}\n"
                      "Time by Step: {timeStep} s\n"
                      "Remaining Time: {time} s\n".format(epReward=round(episode_reward, ndigits=DECIMALS),
                                                      std=round(np.std(actions_taken), ndigits=DECIMALS*2),
                                                      epsilon = round(current_epsilon, ndigits=DECIMALS),
                                                      timeStep=round(np.mean(self.step_time), ndigits=DECIMALS),
                                                      time = round(self.session_time-(time()-start_time), ndigits=DECIMALS)))

            if episodes_between_saving is not None and (episode % episodes_between_saving) == 0:
                self.save()
            if self.session_time-(time()-start_time) < 0:
                self.save()
                if self.verbose: print("Training time finished. Execute again for resuming the train")
                return None

    def additional_maintaing_score(self,composed_state):
        return torch.sum((self.buffer_influence * composed_state[:, ARE_X_Y_VALID_POS])* MAINTAIN_PERSON_BONUS).item()

    def save(self):
        self.current_model.save()
        self.save_losses_rewards_and_buffer(save_data_at=self.save_data_at)
        self.plot_loss_and_rewards(save_at=self.save_data_at)
        if self.executing_on_server and self.send_security_copy:
            # Save a copy on the server
            for file in listdir(self.save_data_at):
                self.socket.send_file(file_path=join(self.save_data_at, file))

    def charge_previous_losses_and_rewards(self, charge_from = RL_CONTROLLER_DIR, losses_file = LOSSES_FILE_NAME,
                                           rewards_file=REWARDS_FILE_NAME,  replay_buffer_file = REPLAY_BUFFER_FILE_NAME):
        losses_path, rewards_path, replay_buffer_path = join(charge_from, losses_file), join(charge_from, rewards_file), \
                                                        join(charge_from, replay_buffer_file)
        losses = load(open(losses_path, 'r'+FILES_CODIFICATION)) if isfile(losses_path) else []
        rewards = load(open(rewards_path, 'r' + FILES_CODIFICATION)) if isfile(rewards_path) else []
        replay_buffer = load(open(replay_buffer_path, 'r' + FILES_CODIFICATION)) if isfile(replay_buffer_path) else ReplayBuffer(capacity=self.buffer_size)
        return losses, rewards, replay_buffer

    def save_losses_rewards_and_buffer(self, save_data_at = RL_CONTROLLER_DIR, losses_file = LOSSES_FILE_NAME,
                                       rewards_file=REWARDS_FILE_NAME, replay_buffer_file = REPLAY_BUFFER_FILE_NAME):
        if not isdir(save_data_at):
            makedirs(save_data_at)
        losses_path, rewards_path, replay_buffer_path = join(save_data_at, losses_file), join(save_data_at, rewards_file), \
                                                        join(save_data_at, replay_buffer_file)
        dump(self.losses,open(losses_path, 'w'+FILES_CODIFICATION))
        dump(self.all_rewards, open(rewards_path, 'w' + FILES_CODIFICATION))
        dump(self.replay_buffer, open(replay_buffer_path, 'w' + FILES_CODIFICATION))


    def plot_loss_and_rewards(self, save_at = RL_CONTROLLER_DIR, smoothness_kernel_shape=SMOOTHNESS_KERNEL_SHAPE, smoothness_sigma=SMOOTHNESS_SIGMA):
        conv_kernel = gaussian(shape=smoothness_kernel_shape, sigma=smoothness_sigma)


        for var, name in ((self.losses, 'Huber Loss'), (self.all_rewards, 'Reward')):
            legend_names = [name, 'Smoothed '+name, 'Peak']
            plt.plot(range(len(var)), var, 'tab:blue', alpha = 0.5)
            smoothness_valid_range = np.arange(smoothness_kernel_shape//2, len(var)-smoothness_kernel_shape//2)
            if len(smoothness_valid_range) > 0:
                spline = np.convolve(var,v=conv_kernel, mode='valid')
                plt.plot(smoothness_valid_range, spline, 'tab:orange')
                peak_arg_fn, peak_fn = (np.argmin, np.min) if 'Loss' in name else (np.argmax, np.max)
                peak_point, peak = peak_arg_fn(spline)+smoothness_kernel_shape//2, peak_fn(spline)
                plt.plot(peak_point, peak, 'ro')
                plt.legend(legend_names)
            plt.title(name+' by Episode')
            plt.xlabel('Episode')
            plt.ylabel(name)
            plt.savefig(join(save_at, name+'.png'))
            plt.close()

def gaussian(shape=11, sigma=1.):
    if shape%2 == 0:
        warn("Gaussian size adjusted to nearest odd: {int1} -> {int2}".format(int1=shape, int2=shape+1))
    vector = np.arange(-shape//2, shape//2, dtype=np.float32)
    vector = 1. / (sqrt(sigma ** pi)) * np.exp(-sigma * (vector*vector))
    # Return the normalized vector
    return vector/np.sum(vector)




