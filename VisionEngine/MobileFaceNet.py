from cv2.dnn import readNetFromCaffe, blobFromImage
from Constants import FACE_EMBEDDING_DEFINITION, FACE_EMBEDDING_WEIGHTS
from cv2 import resize

INPUT_SIZE = (112, 112)
TRAINING_MEAN = 127.5
INVERSE_STD = 0.0078125

class MobileFaceNet:
    def __init__(self):
        """
        Embedder including the MobileFaceNet.
        """
        self.network = readNetFromCaffe(prototxt=FACE_EMBEDDING_DEFINITION,
                                        caffeModel=FACE_EMBEDDING_WEIGHTS)

    def predict(self, face):
        """
        Predicts the embedded representation of the face in the input image with the MobileFaceNet.
        :param face: Numpy. Image containing the face to embed.
        :return: Nunpy of Float. Embedded representation of the input face.
        """
        face = resize(face, INPUT_SIZE)

        blob = blobFromImage(face, scalefactor=INVERSE_STD, mean=TRAINING_MEAN,
                             size=INPUT_SIZE, swapRB=True)
        self.network.setInput(blob)
        face = self.network.forward()[0]
        return face
