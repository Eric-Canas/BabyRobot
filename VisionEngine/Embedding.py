class Embedding:
    def __init__(self, model):
        self.model = model

    def predict(self, face):
        return self.model.predict(face)
