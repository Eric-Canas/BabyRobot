from RobotController.AddOnControllers.Controller import Controller
from RobotController.AddOnControllers.MotorController import MotorController
from RecognitionPipeline import RecognitionPipeline
from RobotController.RLConstants import *
from Constants import KNOWN_NAMES
import numpy as np
from RobotController.MovementControl.FiniteStateMachine import FiniteStateMachine
from time import time

class FiniteStateMachineWorld():
    def __init__(self, objective_person, distance_to_maintain_in_m = DISTANCE_TO_MAINTAIN_IN_CM, wall_security_distance_in_cm = WALL_SECURITY_DISTANCE_IN_M,
                 controller = None, recognition_pipeline = None, average_info_from_n_images = 1, movement_mode=DEFAULT_MOVEMENT_MODE, finite_state_machine=None):
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


    def step(self, time=MOVEMENT_TIME, return_reward = False):
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
        # TODO: Solve the problem with the sensor
        new_state[BACK_DISTANCE_POS] = self.controller.get_back_distance(distance_offset=self.wall_security_distance)
        new_state[FRONT_DISTANCE_POS] = self.controller.get_front_distance(distance_offset=self.wall_security_distance)

        self.finite_state_machine.act(state=new_state)
        if return_reward:
            reward = get_state_reward(state=new_state)
            return reward

    def play(self, plot_reward = False):
        start_time = time()
        play_time = time()-start_time
        while(play_time<PLAY_SESSION_TIME_IN_SECONDS):
            self.step()


    def render(self):
        image = self.controller.capture_image()
        self.recognition_pipeline.show_recognitions(image=image)

def get_state_reward(state):
    y_dist, x_dist, are_x_y_valid, image_difference, back_distance, front_distance = state
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
  # Arduino Map
  return np.clip(a=(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min, a_min=out_min, a_max=out_max)
