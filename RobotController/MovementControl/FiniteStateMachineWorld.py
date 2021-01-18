from RobotController.AddOnControllers.Controller import Controller
from RobotController.AddOnControllers.MotorController import MotorController
from RecognitionPipeline import RecognitionPipeline
from RobotController.RLConstants import *
from Constants import KNOWN_NAMES
import numpy as np
from RobotController.MovementControl.FiniteStateMachine import FiniteStateMachine
from time import time as t
from collections import deque
from matplotlib import pyplot as plt
import os

class FiniteStateMachineWorld():
    def __init__(self, objective_person, distance_to_maintain_in_m = DISTANCE_TO_MAINTAIN_IN_CM, wall_security_distance_in_cm = WALL_SECURITY_DISTANCE_IN_M,
                 controller = None, recognition_pipeline = None, average_info_from_n_images = 1, movement_mode=DEFAULT_MOVEMENT_MODE, finite_state_machine=None):
        """
        Environment. Interacts with the environment and analyzes it. Version for the FiniteState Machine.
       :param objective_person: String. Objective person to follow (Or None if 'follow anyone' mode)
       :param distance_to_maintain_in_m: Float. Distance to maintain with the objective. In Meters
       :param wall_security_distance_in_cm: Float. Minimum distance allowed for a wall.
       :param controller: Controller. Controller of the robot, for communicating with its sensors and actuators.
       :param recognition_pipeline: RecognitionPipeline. Computer Vision Pipeline to use for recognizing the target and
                                                         calculating the distances to it.
       :param average_info_from_n_images: Int. Number of images to average for creating a variable that helps the
                                               to discover when, although it thinks that is in movement, it is
                                               really not.
       :param movement_mode: String. Movement mode. One of: 'sync', 'async', '50-50', '33-66' (Default is 'sync').
       :param finite_state_machine: FiniteStateMachine. Finite State Machine to use.
       """
        if objective_person in KNOWN_NAMES:
            self.objective_person = objective_person
        elif objective_person is None:
            self.objective_person = None
            print("Following anyone (Less secure but faster)")
        else:
            raise ValueError(str(objective_person) + ' is not a known person. Only those ones are known: '+', '.join(KNOWN_NAMES))

        self.distance_to_maintain = distance_to_maintain_in_m
        self.wall_security_distance = wall_security_distance_in_cm
        self.controller = controller if controller is not None else Controller(motor_controller=MotorController(movement_mode=movement_mode))
        self.recognition_pipeline = recognition_pipeline if recognition_pipeline is not None else RecognitionPipeline()
        self.average_info_from_n_images = average_info_from_n_images
        self.movement_mode = movement_mode
        self.finite_state_machine = finite_state_machine if finite_state_machine is not None else FiniteStateMachine(warning_obstacle=self.wall_security_distance)
        self.last_movement_time = t()
        self.motionless_times_detect = deque(maxlen=30)
        self.motionless_times_no_detect = deque(maxlen=30)


    def step(self, time=MOVEMENT_TIME, return_reward = False, verbose=True):
        """
        Checks the state of the world and executes an step in accordance with it.
        :param time: Not used. Included for compatibility.
        :param return_reward: Boolean. If True returns the reward. If False don't returns nothing.
        :param verbose: Boolean. If True, verboses the time with the robot remains motionless.
        :return: If return_reward is True, returns the reward as a Float.
        """
        new_state = np.empty(shape=len(STATES_ORDER), dtype=np.float32)
        image = self.controller.capture_image()
        if self.objective_person is not None:
            distances = [self.recognition_pipeline.get_distance_to_faces(image=image, y_offset=self.distance_to_maintain)
                            for _ in range(self.average_info_from_n_images)]
            distances_to_person = [results[self.objective_person] for results in distances
                                        if self.objective_person in results]
        else:
            distances = [self.recognition_pipeline.get_distance_without_identities(image=image,
                                                                         y_offset=self.distance_to_maintain)
                                                                for _ in range(self.average_info_from_n_images)]
            distances_to_person = [results for results in distances if len(results) > 0]



        if len(distances_to_person) > 0:
            new_state[:Y_DIST_POS + 1] = np.mean(distances_to_person, axis=0)
            new_state[ARE_X_Y_VALID_POS] = 1.
        else:
            new_state[:ARE_X_Y_VALID_POS + 1] = (0., 0., 0.)

        new_state[BACK_DISTANCE_POS] = self.controller.get_back_distance(distance_offset=self.wall_security_distance)
        new_state[FRONT_DISTANCE_POS] = self.controller.get_front_distance(distance_offset=self.wall_security_distance)
        if verbose:
            motionless_time = t()-self.last_movement_time
            if len(distances_to_person) > 0:
                self.motionless_times_detect.append(motionless_time)
            else:
                self.motionless_times_no_detect.append(motionless_time)
            print("Motionless Time Detect Mean: {mean}, STD: {std}".format(mean=round(np.mean(self.motionless_times_detect), ndigits=4),
                                                                    std=round(np.std(self.motionless_times_detect),ndigits=4)))
            print("Motionless Time No Detect Mean: {mean}, STD: {std}".format(
                mean=round(np.mean(self.motionless_times_no_detect), ndigits=4),
                std=round(np.std(self.motionless_times_no_detect), ndigits=4)))

        self.finite_state_machine.act(state=new_state)
        if verbose:
            self.last_movement_time = t()
        if return_reward:
            reward = get_state_reward(state=new_state)
            return reward

    def play(self, plot_reward = True, verbose = True):
        """
        Executes the play bucle.
        :param plot_reward: Boolean. If True, plots the obtained reward every 50 steps.
        :param verbose: Boolean. If True, verboses the time with the robot remains motionless
        """
        start_time = t()
        play_time = t()-start_time
        rewards = []
        actions = 0
        while(play_time<PLAY_SESSION_TIME_IN_SECONDS):
            actions += 1
            rewards.append(self.step(verbose=verbose, return_reward=plot_reward))
            if actions % 50 == 0:
                plot_rewards(rewards=rewards)






    def render(self):
        """
        Shows the image of the environment and the recognitions in it
        """
        image = self.controller.capture_image()
        self.recognition_pipeline.show_recognitions(image=image)

def get_state_reward(state):
    """
        Returns the reward of the environment for the new state
        :param state: List of float. Current state of the environment
        :return: Float. Reward obtained for the new state.
        """
    x_dist, y_dist, are_x_y_valid, image_difference, back_distance, front_distance = state
    # Measure to meters
    if not np.isclose(are_x_y_valid, 0.):
        if y_dist < 0:
            y_dist_reward = map_reward(y_dist, in_min=MAX_ALLOWED_Y_DIST_NEAR, in_max=0.)
        else:
            y_dist_reward = map_reward(y_dist, in_min=MAX_ALLOWED_Y_DIST_FAR, in_max=0.)
        if x_dist < 0:
            x_dist_reward = map_reward(x_dist, in_min=MAX_ALLOWED_X_DIST_LEFT, in_max=0.)
        else:
            x_dist_reward = map_reward(x_dist, in_min=MAX_ALLOWED_X_DIST_RIGHT, in_max=0.)
        dist_to_person_reward = y_dist_reward*Y_DISTANCE_INFLUENCE+x_dist_reward*X_DISTANCE_INFLUENCE
    else:
        dist_to_person_reward = MIN_REWARD_BY_PARAM * LOSE_THE_PERSON_INFLUENCE
    # Remember that in this case, less or equal than 0 is dangerous threshold surpassed
    wall_distance_reward = map_reward(back_distance, in_min=0., in_max=DANGEROUS_WALL_DISTANCE - WALL_SECURITY_DISTANCE_IN_M) \
                           + map_reward(front_distance, in_min=0., in_max=DANGEROUS_WALL_DISTANCE - WALL_SECURITY_DISTANCE_IN_M)
    wall_distance_reward *= WALL_DISTANCE_INFLUENCE/2

    return dist_to_person_reward+wall_distance_reward

def plot_rewards(rewards, path=FINITE_STATE_MACHINE_DIR):
    """
    Plots the obtained rewards.
    :param rewards: List of Float. List with the obtained rewards.
    :param path: String. Path where to save the plot
    :return:
    """
    plt.plot(rewards)
    title = "Average Finite State Machine Reward. Mean - {m}. Std - {std}".format(m=np.round(np.mean(rewards), decimals=2),
                                                                                  std=np.round(np.std(rewards), decimals=2))
    plt.title(title)
    plt.xlabel("Action")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(path, title+'.png'))
    plt.close

def map_reward(x, in_min, in_max, out_min=MIN_REWARD_BY_PARAM, out_max=MAX_REWARD_BY_PARAM):
  """
  Finds the equivalent interpolation of the point x in the 'in' range for the 'out' range.
  :param x: Float. Point in the 'in' range
  :param in_min: Float. Minimum of the 'in' range.
  :param in_max: Float. Maximum of the 'in' range.
  :param out_min: Float. Minimum of the 'out' range.
  :param out_max: Float. Maximum of the 'out' range.
  :return: Float. Wquivalent interpolation of the point x for the 'out' range.
  """
  # Arduino Map
  return np.clip(a=(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, a_min=out_min, a_max=out_max)
