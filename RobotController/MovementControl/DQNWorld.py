from RobotController.AddOnControllers.Controller import Controller
from RobotController.AddOnControllers.MotorController import MotorController
from RecognitionPipeline import RecognitionPipeline
from RobotController.RLConstants import *
from Constants import KNOWN_NAMES
import numpy as np
from time import time as t
from collections import deque


class World():
    def __init__(self, objective_person, distance_to_maintain_in_m = DISTANCE_TO_MAINTAIN_IN_CM, wall_security_distance_in_cm = WALL_SECURITY_DISTANCE_IN_M,
                 controller = None, recognition_pipeline = None, average_info_from_n_images = 1, movement_mode=DEFAULT_MOVEMENT_MODE):
        """
        Environment. Interacts with the environment and analyzes it. Version for the DQN
        :param objective_person: String. Objective person to follow (Or None if 'follow anyone' mode)
        :param distance_to_maintain_in_m: Float. Distance to maintain with the objective. In Meters
        :param wall_security_distance_in_cm: Float. Minimum distance allowed for a wall.
        :param controller: Controller. Controller of the robot, for communicating with its sensors and actuators.
        :param recognition_pipeline: RecognitionPipeline. Computer Vision Pipeline to use for recognizing the target and
                                                          calculating the distances to it.
        :param average_info_from_n_images: Int. Number of images to average for creating a variable that helps the
                                                DQN to discover when, although it thinks that is in movement, it is
                                                really not.
        :param movement_mode: String. Movement mode. One of: 'sync', 'async', '50-50', '33-66' (Default is 'sync').
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
        self.last_image = self.controller.capture_image()
        self.last_movement_time = t()
        self.motionless_times_detect = deque(maxlen=30)
        self.motionless_times_no_detect = deque(maxlen=30)

    def step(self, action, time=MOVEMENT_TIME, verbose=True):
        """
        Execute an action in the environment, checks the new state and calculates the reward.
        :param action: Int. Action to execute.
        :param time: Float. Time between the start of the action execution and the stop
        :param verbose: Boolean. If True, verboses the time spent motionless.
        :return: (List of Float, Float). New state and the reward
        """
        if verbose:
            motionless_time = t()-self.last_movement_time
            print("Motionless Time Detect Mean: {mean}, STD: {std}".format(
                mean=round(np.mean(self.motionless_times_detect), ndigits=4),
                std=round(np.std(self.motionless_times_detect), ndigits=4)))
            print("Motionless Time No Detect Mean: {mean}, STD: {std}".format(
                mean=round(np.mean(self.motionless_times_no_detect), ndigits=4),
                std=round(np.std(self.motionless_times_no_detect), ndigits=4)))
        # Execute action
        if action == IDLE:
            self.controller.idle(time=time)
        elif action == FORWARD:
            self.controller.move_forward(time=time)
        elif action == BACK:
            self.controller.move_back(time=time)
        elif action == CLOCKWISE:
            self.controller.rotate_clockwise(time=time)
        elif action == COUNTER_CLOCKWISE:
            self.controller.rotate_counterclockwise(time=time)
        elif action == LEFT_FRONT:
            self.controller.go_left_front(time=time)
        elif action == RIGHT_FRONT:
            self.controller.go_right_front(time=time)
        elif action == LEFT_BACK:
            self.controller.go_left_back(time=time)
        elif action == RIGHT_BACK:
            self.controller.go_right_back(time=time)
        elif action == HALF_LEFT_FRONT:
            self.controller.go_left_front(time=time/2)
        elif action == HALF_RIGHT_FRONT:
            self.controller.go_right_front(time=time/2)
        elif action == HALF_LEFT_BACK:
            self.controller.go_left_back(time=time/2)
        elif action == HALF_RIGHT_BACK:
            self.controller.go_right_back(time=time/2)
        else:
            raise ValueError("Action {act} does not exist".format(act=action))

        if verbose:
            self.last_movement_time = t()
        # Discover new state
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

        if verbose:
            if len(distances_to_person) > 0:
                self.motionless_times_detect.append(motionless_time)
            else:
                self.motionless_times_no_detect.append(motionless_time)

        if len(distances_to_person) > 0:
            new_state[:Y_DIST_POS + 1] = np.mean(distances_to_person, axis=0)
            new_state[ARE_X_Y_VALID_POS] = 1.
        else:
            new_state[:ARE_X_Y_VALID_POS + 1] = (0., 0., 0.)
        new_state[IMAGE_DIFFERENCE_POS] = np.mean(abs(image-self.last_image))/255
        # TODO: Solve the problem with the sensor
        new_state[BACK_DISTANCE_POS] = self.controller.get_back_distance(distance_offset=self.wall_security_distance)
        new_state[FRONT_DISTANCE_POS] = self.controller.get_front_distance(distance_offset=self.wall_security_distance)

        reward = get_state_reward(state=new_state)
        return new_state, reward


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
