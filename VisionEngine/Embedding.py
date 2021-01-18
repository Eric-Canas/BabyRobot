class Embedding:
    def __init__(self, model):
        """
        General class for executing an embedding model
        :param model: Usually FlattenEmbedding or MobileFaceNet. Embedding Model.
        """
        self.model = model

    def predict(self, face):
        """
        Executes a prediction with the model.
        :param face: Numpy. Input image of the face.
        :return: Numpy of Float. Embedded representation of the Input image.
        """
        return self.model.predict(face)
