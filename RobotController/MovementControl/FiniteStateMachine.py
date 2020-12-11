from random import random, choice
from RobotController.RLConstants import *
import numpy as np
from RobotController.AddOnControllers.Controller import Controller
from RobotController.AddOnControllers.MotorController import MotorController

APPROACHING, ESCAPING, TURNING, CENTERED, CATCHING_ATTENTION, SEARCHING, AVOIDING_OBSTACLE = 1, 2, 3, 4, 5, 6, 7
STATES = {APPROACHING : 'Approaching', ESCAPING: 'Escaping', TURNING: 'Turning', CENTERED: 'Centered',
          CATCHING_ATTENTION:'Catching Attention', SEARCHING: 'Searching', AVOIDING_OBSTACLE : 'Avoiding Obstacle'}

INACCURATE_LOCATION_GROUP = {APPROACHING, ESCAPING, TURNING}
ACCURATE_LOCATION_GROUP = {CENTERED, CATCHING_ATTENTION}
BABY_LOCATION_IS_KNOWN_GROUP = {APPROACHING, ESCAPING, TURNING, CENTERED, CATCHING_ATTENTION}
LEFT, RIGHT = -1, 1

class FiniteStateMachine:
    def __init__(self, controller=None, movement_mode=DEFAULT_MOVEMENT_MODE, dist_epsilon=DIST_EPSILON,
                 catching_attention_prob=CATCHING_ATTENTION_PROB):
        self.state = SEARCHING
        self.dist_epsilon = dist_epsilon
        self.catching_attention_prob = catching_attention_prob
        self.controller = controller if controller is not None else Controller(motor_controller=MotorController(movement_mode=movement_mode))
        self.last_search_direction = None

    def act(self, state, verbose = True):
        y_dist, x_dist, are_x_y_valid, image_difference, back_distance, front_distance = state
        are_x_y_valid = not np.isclose(are_x_y_valid, 0)
        x_deviation, y_deviation = self.location_deviation(dist=x_dist), self.location_deviation(dist=y_dist)

        # ----------- TRANSITIONS SUMMARIZED ------------
        # OBSTACLE APPEARED
        if back_distance < 0 or front_distance < 0:
            self.state = AVOIDING_OBSTACLE
        # LOST TARGET TRANSITION
        elif not are_x_y_valid:
            if self.state != SEARCHING:
                # TODO: Improve the strategy by taking into account the last place where target was seen
                self.last_search_direction = choice((LEFT, RIGHT))
            self.state = SEARCHING
        # RELOCATION
        else:
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
            self.turn_to_x(x_dist=x_dist)

        elif self.state == CENTERED:
            self.controller.idle()

        elif self.state == CATCHING_ATTENTION:
            self.catch_attention()

        elif self.state == SEARCHING:
            self.search()

        elif self.state == AVOIDING_OBSTACLE:
            self.avoid_obstacle(back_distance=back_distance, front_distance=front_distance)

    def location_deviation(self, dist):
        if np.isclose(dist,0, rtol=self.dist_epsilon):
            return 0
        elif dist < dist-self.dist_epsilon:
            return -1
        else:
            return 1

    def get_best_innacurate_state(self, x_dev, y_dev):
        if y_dev < 0:
            return ESCAPING
        if x_dev != 0:
            return TURNING
        else:
            return APPROACHING

    def turn_to_x(self, x_dist):
        time = map(x=abs(x_dist), in_min=0, in_max=45, out_min=0, out_max=MOVEMENT_TIME)
        if x_dist < 0:
            self.controller.go_left_back(time=time)
        else:
            self.controller.go_right_back(time=time)

    def catch_attention(self):
        self.controller.rotate_clockwise()
        self.controller.rotate_counterclockwise()
        self.controller.rotate_counterclockwise()
        self.controller.rotate_clockwise()

    def search(self):
        if self.last_search_direction == RIGHT:
            self.controller.rotate_clockwise()
        else:
            self.controller.rotate_counterclockwise()

    def avoid_obstacle(self, back_distance, front_distance):
        pass

def map(x, in_min, in_max, out_min, out_max):
  # Arduino Map
  return np.clip(a=(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, a_min=out_min, a_max=out_max)

