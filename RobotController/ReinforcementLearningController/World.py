from RobotController.AddOnControllers.Controller import Controller
from RecognitionPipeline import RecognitionPipeline
from RobotController.RLConstants import *
from Constants import KNOWN_NAMES
import numpy as np


class World():
    def __init__(self, objective_person, distance_to_maintain_in_m = DISTANCE_TO_MAINTAIN_IN_M, back_security_distance_in_cm = BACK_SECURITY_DISTANCE_IN_M,
                 controller = None, recognition_pipeline = None, average_info_from_n_images = 1):
        if objective_person in KNOWN_NAMES:
            self.objective_person = objective_person
        elif objective_person is None:
            self.objective_person = None
            print("Following anyone (Less secure but faster)")
        else:
            raise ValueError(str(objective_person) + ' is not a known person. Only those ones are known: '+', '.join(KNOWN_NAMES))

        self.distance_to_maintain = distance_to_maintain_in_m
        self.back_security_distance = back_security_distance_in_cm
        self.controller = controller if controller is not None else Controller()
        self.recognition_pipeline = recognition_pipeline if recognition_pipeline is not None else RecognitionPipeline()
        self.average_info_from_n_images = average_info_from_n_images


    def step(self, action, time=None):
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
        elif action == LEFT:
            self.controller.go_left(time=time)
        elif action == RIGHT:
            self.controller.go_right(time=time)

        # Discover new state
        new_state = np.empty(shape=len(STATES_ORDER), dtype=np.float32)
        if self.objective_person is not None:
            distances = [self.recognition_pipeline.get_distance_to_faces(image=self.controller.capture_image(), y_offset=self.distance_to_maintain)
                            for _ in range(self.average_info_from_n_images)]
            distances_to_person = [results[self.objective_person] for results in distances
                                        if self.objective_person in results]
        else:
            distances = [self.recognition_pipeline.get_distance_without_identities(image=self.controller.capture_image(),
                                                                         y_offset=self.distance_to_maintain)
                                                                for _ in range(self.average_info_from_n_images)]
            distances_to_person = [results for results in distances if len(results) > 0]


        if len(distances_to_person) > 0:
            new_state[:Y_DIST_POS + 1] = np.mean(distances_to_person, axis=0)
            new_state[ARE_X_Y_VALID_POS] = 1.
        else:
            new_state[:ARE_X_Y_VALID_POS + 1] = (0., 0., 0.)

        # TODO: Solve the problem with the sensor
        new_state[BACK_DISTANCE_POS] = self.controller.get_back_distance(back_distance_offset=self.back_security_distance)

        reward = get_state_reward(state=new_state)
        return new_state, reward


    def render(self):
        image = self.controller.capture_image()
        self.recognition_pipeline.show_detections(image=image)

def get_state_reward(state):
    y_dist, x_dist, are_x_y_valid, back_distance = state

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
    back_distance_reward = map_reward(back_distance, in_min=0., in_max=DANGEROUS_BACK_DISTANCE-BACK_SECURITY_DISTANCE_IN_M)
    back_distance_reward *= BACK_DISTANCE_INFLUENCE

    return dist_to_person_reward+back_distance_reward

def map_reward(x, in_min, in_max, out_min=MIN_REWARD_BY_PARAM, out_max=MAX_REWARD_BY_PARAM):
  # Arduino Map
  return np.clip(a=(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, a_min=out_min, a_max=out_max)
