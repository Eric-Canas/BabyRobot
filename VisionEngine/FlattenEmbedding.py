from cv2 import resize, cvtColor, COLOR_BGR2GRAY

INPUT_SIZE = (64, 64)

class FlattenEmbedding:
    def __init__(self):
        self.network = None

    def predict(self, face):
        face = resize(face, INPUT_SIZE)
        face = cvtColor(face, COLOR_BGR2GRAY)
        return face.flatten()
