from cv2.dnn import readNetFromCaffe, blobFromImage
from Constants import *
from cv2 import resize, cvtColor, COLOR_BGR2GRAY

INPUT_SIZE = (64, 64)
TRAINING_MEAN = 127.5
INVERSE_STD = 0.0078125

class FlattenEmbedding:
    def __init__(self):
        self.network = None

    def predict(self, face):
        face = resize(face, INPUT_SIZE)
        face = cvtColor(face, COLOR_BGR2GRAY)
        return face.flatten()
