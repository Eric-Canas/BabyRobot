from os.path import join

INVALID = -1

DQN_REPLAY_BUFFER_CAPACITY = 1000
DQN_BATCH_SIZE = 64

IDLE, FORWARD, BACK, COUNTER_CLOCKWISE, CLOCKWISE, RIGHT_FRONT, LEFT_FRONT, RIGHT_BACK, LEFT_BACK, HALF_RIGHT_FRONT, HALF_LEFT_FRONT, HALF_RIGHT_BACK, HALF_LEFT_BACK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

ACTIONS_DEFINITION = {IDLE: 'IDLE', FORWARD: 'GO FORWARD', BACK: 'GO BACK', COUNTER_CLOCKWISE: 'TURN COUNTER_CLOCKWISE',
                      CLOCKWISE: 'TURN COUNTER COUNTER_CLOCKWISE', RIGHT_FRONT: 'GO RIGHT FRONT', LEFT_FRONT: 'GO LEFT FRONT',
                      RIGHT_BACK: 'GO RIGHT BACK', LEFT_BACK : 'GO LEFT BACK', HALF_RIGHT_FRONT: 'GO HALF RIGHT FRONT', HALF_LEFT_FRONT: 'GO HALF LEFT FRONT',
                      HALF_RIGHT_BACK: 'GO HALF RIGHT BACK', HALF_LEFT_BACK : 'GO HALF LEFT BACK',}

ACTIONS_TELEOPERATED_KEYPAD_DEFINITON = {5 : IDLE, 8 : FORWARD, 2 : BACK, 4 : COUNTER_CLOCKWISE,
                                         6 : CLOCKWISE, 9 : RIGHT_FRONT, 7 : LEFT_FRONT, 3 : RIGHT_BACK, 1 : LEFT_BACK}

# (Degrees rotated: 1 max (45º), 0 No movement (0º), -1 min (-45º), Advance: 1 front, 0 no movement, -1 back)
ROTATION_ADVANCE_BY_ACTION = {IDLE: (0.,0.), FORWARD: (0., 1.), BACK: (0.,-1.), COUNTER_CLOCKWISE: (-1,0.), CLOCKWISE: (1, 0),
                              RIGHT_FRONT: (0.5, 0.25), LEFT_FRONT: (-0.5, 0.25), RIGHT_BACK: (0.5, -0.25), LEFT_BACK: (-0.5,-0.25),
                              HALF_RIGHT_FRONT: (0.5/2, 0.25/2), HALF_LEFT_FRONT: (-0.5/2, 0.25/2), HALF_RIGHT_BACK: (0.5/2, -0.25/2),
                              HALF_LEFT_BACK: (-0.5/2,-0.25/2),
                              -1 : (0., 0.)}

X_DIST_POS, Y_DIST_POS, ARE_X_Y_VALID_POS, IMAGE_DIFFERENCE_POS, BACK_DISTANCE_POS, FRONT_DISTANCE_POS = 0, 1, 2, 3, 4, 5

STATES_ORDER = [X_DIST_POS, Y_DIST_POS, ARE_X_Y_VALID_POS, IMAGE_DIFFERENCE_POS, BACK_DISTANCE_POS, FRONT_DISTANCE_POS]

INPUT_LAST_ACTIONS = 12


ACTIONS_PER_TRAIN_STEP = 5
DQN_LEARNING_RATE = 0.001
GAMMA = 0.9
TRAIN_STEPS = 10000
DQN_UPDATE_RATIO = 25
EPSILON_START, EPSILON_FINAL, EPSILON_DECAY = 1., 0.1, 0.85
STEPS_PER_EPISODE = 76
EPISODES_BETWEEN_SAVING = 5
USUAL_CASE_USE_PROB = 0.05

RL_CONTROLLER_DIR = join('Models', 'RLController')
RL_CONTROLLER_PTH_FILE = 'Controller.pth'

LOSSES_FILE_NAME = 'CumulatedLosses.pkl'
REWARDS_FILE_NAME = 'CumulatedReward.pkl'
REPLAY_BUFFER_FILE_NAME = 'LastReplayBuffer.pkl'

FILES_CODIFICATION = 'b'

SMOOTHNESS_KERNEL_SHAPE = 15
SMOOTHNESS_SIGMA = 0.5

DISTANCE_TO_MAINTAIN_IN_CM = 75 # m
WALL_SECURITY_DISTANCE_IN_M = 0.3
SYNC_MODE, ASYNC_MODE, HALF_MODE, ONE_THIRD_MODE = 'sync', 'async', '50-50', '33-66'
DEFAULT_MOVEMENT_MODE = SYNC_MODE
# Each Parameter influence

Y_DISTANCE_INFLUENCE, X_DISTANCE_INFLUENCE, WALL_DISTANCE_INFLUENCE = .6, .1, .3
LOSE_THE_PERSON_INFLUENCE = Y_DISTANCE_INFLUENCE + X_DISTANCE_INFLUENCE - 0.05
MAX_REWARD_BY_PARAM, MIN_REWARD_BY_PARAM = 10., 0. # Only using positive values for favouring the DQN training using ReLUs
MAX_ALLOWED_Y_DIST_NEAR, MAX_ALLOWED_Y_DIST_FAR = -DISTANCE_TO_MAINTAIN_IN_CM // 2, DISTANCE_TO_MAINTAIN_IN_CM * 4
MAX_ALLOWED_X_DIST_LEFT, MAX_ALLOWED_X_DIST_RIGHT = -100, 100
MAINTAIN_PERSON_BONUS = 4

DANGEROUS_WALL_DISTANCE = WALL_SECURITY_DISTANCE_IN_M * 2
IMPROVEMENT_BONUS = 0.1
PLAY_SESSION_TIME_IN_SECONDS = 9*60*60
MOVEMENT_TIME = 0.6 # Aproximately the time for which rotation is of 90 degrees and turn is of 45 degrees
REQUEST_FOR_ACTION_TIMEOUT = 1
SAVE_NON_REWARDED_STATE_PROB = 0.05

DIST_EPSILON = 15
CATCHING_ATTENTION_PROB = 0.05
ENSURE_LOSE_IMAGES = 5
CONSECUTIVE_AVOIDING_OBSTACLES_TRIES = 5