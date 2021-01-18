from cv2 import resize, cvtColor, COLOR_BGR2GRAY

INPUT_SIZE = (64, 64)

class FlattenEmbedding:
    def __init__(self):
        """
        Simple embedding that only transforms the image to grayscale and flattens it.
        """
        self.network = None

    def predict(self, face):
        """
        Returns a flatten grayscale representation of the image
        :param face: Numpy. Input image containing the face to embed.
        :return: Numpy of Float. Flatten grayscale representation of the image
        """
        face = resize(face, INPUT_SIZE)
        face = cvtColor(face, COLOR_BGR2GRAY)
        return face.flatten()
