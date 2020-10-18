from os.path import join


DQN_REPLAY_BUFFER_CAPACITY = 1000
DQN_BATCH_SIZE = 64

IDLE, FORWARD, BACK, LEFT, RIGHT = 0, 1, 2, 3, 4
ACTIONS_DEFINITION = {IDLE: 'IDLE', FORWARD: 'GO FORWARD', BACK: 'GO BACK', LEFT: 'GO LEFT', RIGHT: 'GO RIGHT'}
X_DISTANCE_POS, Y_DISTANCE_POS, ARE_X_Y_VALID_POS, BACK_DISTANCE_POS = 0, 1, 2, 3
STATES_ORDER = [X_DISTANCE_POS, Y_DISTANCE_POS, ARE_X_Y_VALID_POS, BACK_DISTANCE_POS]



ACTIONS_PER_TRAIN_STEP = 5
GAMMA = 0.99
TRAIN_STEPS = 10000
DQN_UPDATE_RATIO = 50
EPSILON_START, EPSILON_FINAL, EPSILON_DECAY = 1., 0.02, 10
STEPS_PER_EPISODE = 20
EPISODES_BETWEEN_SAVING = 5

RL_CONTROLLER_DIR = join('Models', 'RLController')
RL_CONTROLLER_PTH_FILE = 'Controller.pth'

LOSSES_FILE_NAME = 'CumulatedLosses.pkl'
REWARDS_FILE_NAME = 'CumulatedReward.pkl'

FILES_CODIFICATION = 'b'

SMOOTHNESS_KERNEL_SHAPE = 15
SMOOTHNESS_SIGMA = 0.5

DISTANCE_TO_MAINTAIN_IN_M = 1 # m
BACK_SECURITY_DISTANCE_IN_M = 0.25

# Each Parameter influence

Y_DISTANCE_INFLUENCE, X_DISTANCE_INFLUENCE, BACK_DISTANCE_INFLUENCE = .4, .1, .5
LOSE_THE_PERSON_INFLUENCE = Y_DISTANCE_INFLUENCE + X_DISTANCE_INFLUENCE - 0.05
MAX_REWARD_BY_PARAM, MIN_REWARD_BY_PARAM = 10., 0. # Only using positive values for favouring the DQN training using ReLUs
MAX_ALLOWED_Y_DIST_NEAR, MAX_ALLOWED_Y_DIST_FAR = -DISTANCE_TO_MAINTAIN_IN_M//2, DISTANCE_TO_MAINTAIN_IN_M*4
MAX_ALLOWED_X_DIST_LEFT, MAX_ALLOWED_X_DIST_RIGHT = -10, 10
DANGEROUS_BACK_DISTANCE = BACK_SECURITY_DISTANCE_IN_M*2

