from random import random, choice
from RobotController.RLConstants import *
import numpy as np
from RobotController.AddOnControllers.Controller import Controller
from RobotController.AddOnControllers.MotorController import MotorController
from VisionEngine.OpenCVFaceDetector import INPUT_SIZE
APPROACHING, ESCAPING, TURNING, CENTERED, CATCHING_ATTENTION, SEARCHING, AVOIDING_OBSTACLE = 1, 2, 3, 4, 5, 6, 7
STATES = {APPROACHING : 'Approaching', ESCAPING: 'Escaping', TURNING: 'Turning', CENTERED: 'Centered',
          CATCHING_ATTENTION:'Catching Attention', SEARCHING: 'Searching', AVOIDING_OBSTACLE : 'Avoiding Obstacle'}

INACCURATE_LOCATION_GROUP = {APPROACHING, ESCAPING, TURNING}
ACCURATE_LOCATION_GROUP = {CENTERED, CATCHING_ATTENTION}
BABY_LOCATION_IS_KNOWN_GROUP = {APPROACHING, ESCAPING, TURNING, CENTERED, CATCHING_ATTENTION}
LEFT, RIGHT = -1, 1
NEAR, FAR = -1, 1
NO_DEVIATION = 0

class FiniteStateMachine:
    def __init__(self, controller=None, movement_mode=DEFAULT_MOVEMENT_MODE, dist_epsilon=DIST_EPSILON,
                 catching_attention_prob=CATCHING_ATTENTION_PROB, ensure_lose_images=ENSURE_LOSE_IMAGES,
                 consecutive_avoiding_obstacles_tries = CONSECUTIVE_AVOIDING_OBSTACLES_TRIES):
        """
        Finite State Machine model for controlling the robot.
        :param controller: Controller. Controller of the robot, for communicating with its sensors and actuators.
        :param movement_mode: String. Movement mode. One of: 'sync', 'async', '50-50', '33-66' (Default is 'sync').
        :param dist_epsilon: Float. Maximum allowed error for considering that the target is not centered.
        :param catching_attention_prob: Float. Probability of entering in the state 'Catch Attention' the machine is
                                               in the state 'Centered'.
        :param ensure_lose_images: Int. Amount of images that must be taken without detecting the target for considering
                                        that it is lost.
        :param consecutive_avoiding_obstacles_tries: Int. Maximum tries for avoiding an obstacle for considering that
                                                          it is a wall.
        """

        self.state = SEARCHING
        self.dist_epsilon = dist_epsilon
        self.catching_attention_prob = catching_attention_prob
        self.controller = controller if controller is not None else Controller(motor_controller=MotorController(movement_mode=movement_mode))
        self.last_seen_direction = None
        self.ensure_lose_images = ensure_lose_images
        self.consecutive_losed_images = 0
        self.last_x_deviation, self.last_y_deviation = 0, 0
        self.consecutive_avoiding_obstacles = 0
        self.consecutive_avoiding_obstacles_tries = consecutive_avoiding_obstacles_tries

    def act(self, state, verbose = True):
        """
        Checks the environment, updates the state and executes the action associated to the new state.
        :param state: String. Current state of the environment
        :param verbose: Boolean. If True, verboses the process
        """
        x_dist, y_dist, are_x_y_valid, back_distance, front_distance = state[X_DIST_POS], state[Y_DIST_POS], state[ARE_X_Y_VALID_POS], state[BACK_DISTANCE_POS], state[FRONT_DISTANCE_POS]
        are_x_y_valid = not np.isclose(are_x_y_valid, 0)
        x_deviation, y_deviation = self.x_location_deviation(dist=x_dist), self.y_location_deviation(dist=y_dist)
        # ----------- TRANSITIONS SUMMARIZED ------------
        # OBSTACLE APPEARED
        if back_distance < 0 or front_distance < 0:
            self.state = AVOIDING_OBSTACLE
            self.consecutive_avoiding_obstacles += 1
        # LOST TARGET TRANSITION
        elif not are_x_y_valid:
            self.consecutive_avoiding_obstacles = 0
            if self.state != SEARCHING:
                # TODO: Improve the strategy by taking into account the last place where target was seen
                self.last_seen_direction = self.last_x_deviation
                self.consecutive_losed_images = 0

            self.consecutive_losed_images += 1
            if verbose:
                print("Losed images: {l}".format(l=self.consecutive_losed_images))
            self.state = SEARCHING
        # RELOCATION
        else:
            self.consecutive_avoiding_obstacles = 0
            if x_deviation == 0 and y_deviation == 0:
                self.state = CATCHING_ATTENTION if random() <= self.catching_attention_prob else CENTERED
            else:
                self.state = self.get_best_innacurate_state(x_dev=x_deviation, y_dev=y_deviation)

        if verbose:
            print('STATE: {state}'.format(state=STATES[self.state]))

        # ----------------- ACTIONS ------------------
        if self.state == APPROACHING:
            self.controller.move_forward()

        elif self.state == ESCAPING:
            self.controller.move_back()

        elif self.state == TURNING:
            self.turn_to_x(x_dist=x_dist, y_dist=y_dist)

        elif self.state == CENTERED:
            self.controller.idle()

        elif self.state == CATCHING_ATTENTION:
            self.catch_attention()

        elif self.state == SEARCHING:
            if self.consecutive_losed_images > self.ensure_lose_images:
                self.search()

        elif self.state == AVOIDING_OBSTACLE:
            self.avoid_obstacle(back_distance=back_distance, front_distance=front_distance)
        if are_x_y_valid:
            self.last_x_deviation = x_deviation

    def x_location_deviation(self, dist):
        """
        Returns -1 if the target is relevantly deviated to left, 1 if it is relevantly deviated to right or 0 if it is
        not relevantly deviated.
        :param dist: Float. X distance to the target.
        :return: Int. -1 if the target is relevantly deviated to left, 1 if it is relevantly deviated to right
                    or 0 if it is not relevantly deviated.
        """
        if -self.dist_epsilon*2 < dist < self.dist_epsilon*2:
            return NO_DEVIATION
        elif dist < -self.dist_epsilon*2:
            return LEFT
        else:
            return RIGHT

    def y_location_deviation(self, dist):
        """
        Returns -1 if the target is relevantly too close, 1 if it is too far or 0 otherwise
        :param dist: Float. Y distance to the target.
        :return: Int. -1 if the target is relevantly too close, 1 if it is too far or 0 otherwise
        """
        if -self.dist_epsilon < dist < self.dist_epsilon:
            return NO_DEVIATION
        elif dist < -self.dist_epsilon:
            return NEAR
        else:
            return FAR


    def get_best_innacurate_state(self, x_dev, y_dev):
        """
        Gets which is the best state for cases where the target is located but not centered.
        :param x_dev: Int. X Deviation of the target.
        :param y_dev: Int. Y Deviation of the target.
        :return:
        """
        if y_dev < 0:
            return ESCAPING
        if x_dev != 0:
            return TURNING
        else:
            return APPROACHING

    def turn_to_x(self, x_dist, y_dist):
        """
        Turns to the direction to which the target is deviated, in order to re-center it. The amount of time turning
        depends how much deviated it is. If it turns advancing at the same time of retreating depends on the y distance.
        :param x_dist: Float. X Distance to the target.
        :param y_dist: Float. Y Distance to the target.
        """
        time = map(x=abs(x_dist), in_min=0, in_max=INPUT_SIZE[-1]/2, out_min=0, out_max=MOVEMENT_TIME/1.75)
        if x_dist < 0.:
            if y_dist <= 0.:
                self.controller.go_left_back(time=time)
            else:
                self.controller.go_left_front(time=time)
        else:
            if y_dist <= 0.:
                self.controller.go_right_back(time=time)
            else:
                self.controller.go_right_front(time=time)

    def catch_attention(self):
        """
        Executes the 'Catch Attention' action. It is... rotate clockwise 90º, rotate counterclokwise 180º and
        rotate clockwise 90º again, in order to return to the initial position.
        :return:
        """
        self.controller.rotate_clockwise()
        self.controller.rotate_counterclockwise()
        self.controller.rotate_counterclockwise()
        self.controller.rotate_clockwise()

    def search(self):
        """
        Rotates to the direction to which the target was seen by last time before lost.
        """
        if self.last_seen_direction == RIGHT:
            self.controller.rotate_clockwise(time=MOVEMENT_TIME/2)
        else:
            self.controller.rotate_counterclockwise(time=MOVEMENT_TIME/2)

    def avoid_obstacle(self, back_distance, front_distance):
        """
        Executes the movement for avoiding an obstacle.
        :param back_distance: Float. Distance to the closest obstacle in the back.
        :param front_distance: Float. Distance to the closest obstacle in the front.
        :return:
        """
        escape_direction = choice([RIGHT, LEFT])
        if self.consecutive_avoiding_obstacles > self.consecutive_avoiding_obstacles:
            for _ in range(4):
                self.controller.rotate_clockwise() if escape_direction == RIGHT \
                    else self.controller.rotate_counterclockwise()
            self.consecutive_avoiding_obstacles = self.consecutive_avoiding_obstacles//2
        elif back_distance < 0 and front_distance < 0:
            self.controller.rotate_clockwise() if escape_direction == RIGHT \
                            else self.controller.rotate_counterclockwise()
            self.controller.move_forward()

        elif back_distance < 0:
            self.controller.go_right_front() if escape_direction == RIGHT \
                else self.controller.go_left_front()
            if self.controller.get_front_distance() < 0:
                self.controller.move_forward()
            elif self.controller.get_back_distance() < self.dist_epsilon:
                self.controller.move_back()
        elif front_distance < 0:
            self.controller.go_right_back() if escape_direction == RIGHT \
                else self.controller.go_left_back()
            if self.controller.get_back_distance() < 0:
                self.controller.move_back()
            elif self.controller.get_front_distance() < self.dist_epsilon:
                self.controller.move_forward()


def map(x, in_min, in_max, out_min, out_max):
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

