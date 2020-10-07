from cv2.dnn import readNetFromTensorflow
from Constants import *

INPUT_SIZE = (112, 112)

class Embedding:
    def __init__(self, model):
        self.model = model

    def predict(self, face):
        return self.model.predict(face)
