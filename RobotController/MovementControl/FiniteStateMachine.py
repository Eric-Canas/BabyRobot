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
NO_DEVIATION = 0

class FiniteStateMachine:
    def __init__(self, controller=None, movement_mode=DEFAULT_MOVEMENT_MODE, dist_epsilon=DIST_EPSILON,
                 catching_attention_prob=CATCHING_ATTENTION_PROB, ensure_lose_images=ENSURE_LOSE_IMAGES):
        self.state = SEARCHING
        self.dist_epsilon = dist_epsilon
        self.catching_attention_prob = catching_attention_prob
        self.controller = controller if controller is not None else Controller(motor_controller=MotorController(movement_mode=movement_mode))
        self.last_seen_direction = None
        self.ensure_lose_images = ensure_lose_images
        self.consecutive_losed_images = 0
        self.last_x_deviation, self.last_y_deviation = 0, 0

    def act(self, state, verbose = True):
        x_dist, y_dist, are_x_y_valid, back_distance, front_distance = state[X_DIST_POS], state[Y_DIST_POS], state[ARE_X_Y_VALID_POS], state[BACK_DISTANCE_POS], state[FRONT_DISTANCE_POS]
        are_x_y_valid = not np.isclose(are_x_y_valid, 0)
        x_deviation, y_deviation = self.location_deviation(dist=x_dist), self.location_deviation(dist=y_dist)
        if verbose:
            print("X: {x_dist}, Y: {y_dist} \n "
                  "X DEV: {x_dev}, Y DEV: {y_dev} \n"
                  "BACK OBSTACLE:{back}, FRONT OBSTACLE: {front}".format(x_dist=x_dist, y_dist=y_dist,
                                                                         x_dev=x_deviation, y_dev=y_deviation,
                                                                         back=back_distance, front=front_distance))
        # ----------- TRANSITIONS SUMMARIZED ------------
        # OBSTACLE APPEARED
        if back_distance < 0 or front_distance < 0:
            self.state = AVOIDING_OBSTACLE
        # LOST TARGET TRANSITION
        elif not are_x_y_valid:
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

        self.last_x_deviation = x_deviation

    def location_deviation(self, dist):
        if -self.dist_epsilon < dist < self.dist_epsilon:
            return NO_DEVIATION
        elif dist < -self.dist_epsilon:
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

    def turn_to_x(self, x_dist, y_dist):
        time = map(x=abs(x_dist), in_min=0, in_max=INPUT_SIZE[-1]/2, out_min=0, out_max=MOVEMENT_TIME)
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
        self.controller.rotate_clockwise()
        self.controller.rotate_counterclockwise()
        self.controller.rotate_counterclockwise()
        self.controller.rotate_clockwise()

    def search(self):
        if self.last_seen_direction == RIGHT:
            self.controller.rotate_clockwise(time=MOVEMENT_TIME/2)
        else:
            self.controller.rotate_counterclockwise(time=MOVEMENT_TIME/2)

    def avoid_obstacle(self, back_distance, front_distance):
        pass

def map(x, in_min, in_max, out_min, out_max):
  # Arduino Map
  return np.clip(a=(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, a_min=out_min, a_max=out_max)

