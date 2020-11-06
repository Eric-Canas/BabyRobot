import os
from numpy import sort
FULL_PHOTOS_DATA_DIR = os.path.join('Dataset', 'CompletePhotos')
CROPPED_FACES_DATA_DIR = os.path.join('Dataset', 'Faces')
NEW_PROPOSALS_DATA_DIR = 'CapturedFromPiCamera'

KNOWN_NAMES = sort(os.listdir(CROPPED_FACES_DATA_DIR))

DEFAULT_PERSON_TO_FOLLOW = 'Eric'

FACE_DETECTOR_DIR = os.path.join('Models', 'FaceDetector')
FACE_EMBEDDING_DIR = os.path.join('Models', 'FaceEmbedding')
FACE_RECOGNIZER_DIR = os.path.join('Models', 'FaceRecognizer')
BODY_DETECTOR_DIR = os.path.join('Models', 'BodyDetection')

FACE_RECOGNIZER_FILE = 'recognizer.pkl'
FACE_RECOGNIZER_VAL_CONFUSION_MATRIX = os.path.join(FACE_RECOGNIZER_DIR, 'val_confusion_matrix.pkl')

FACE_DETECTOR_DEFINITION = os.path.join(FACE_DETECTOR_DIR, 'deploy.prototxt')
FACE_DETECTOR_WEIGHTS = os.path.join(FACE_DETECTOR_DIR, 'weights.caffemodel')

FACE_EMBEDDING_DEFINITION = os.path.join(FACE_EMBEDDING_DIR, 'MobileFaceNet.prototxt')
FACE_EMBEDDING_WEIGHTS = os.path.join(FACE_EMBEDDING_DIR, 'MobileFaceNet.caffemodel')

BODY_DETECTION_DEFINITION = os.path.join(BODY_DETECTOR_DIR, 'Pelee.prototxt')
BODY_DETECTION_WEIGHTS = os.path.join(BODY_DETECTOR_DIR, 'Pelee.caffemodel')

NEURAL_NET_INPUT_SIZE = (300,300)
NEURAL_NET_INPUT_WIDTH = 400

MIN_CONFIDENCE = 0.5

BGR_BLUE = (255,0,0)
DECIMALS = 2
RECTANGLES_THICKNESS = 3
LETTERS_SIZE = 2
CHARGED_WIDTH = 512

FACE_HEIGHT = 12 #For the picture that we are using for training
FACE_WIDTH = 8