from cv2.dnn import readNetFromCaffe, blobFromImage
from Constants import *
from cv2 import resize

INPUT_SIZE = (112, 112)
TRAINING_MEAN = 127.5
INVERSE_STD = 0.0078125

class MobileFaceNet:
    def __init__(self):
        self.network = readNetFromCaffe(prototxt=FACE_EMBEDDING_DEFINITION,
                                        caffeModel=FACE_EMBEDDING_WEIGHTS)

    def predict(self, face):
        face = resize(face, INPUT_SIZE)

        blob = blobFromImage(face, scalefactor=INVERSE_STD, mean=TRAINING_MEAN,
                             size=INPUT_SIZE, swapRB=True)
        self.network.setInput(blob)
        face = self.network.forward()[0]
        return face
