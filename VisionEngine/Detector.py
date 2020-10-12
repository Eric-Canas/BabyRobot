from cv2 import cvtColor, COLOR_BGR2RGB


class Detector:
    def __init__(self, model):
        self.model = model
        self.camera_calculator = ()

    def predict(self, image):
        return self.model.predict(image)

    def crop_boxes_content(self, image, boxes=None, as_rgb=False):
        if boxes is None:
            boxes = self.predict(image=image)
        faces = [image[y1:y2, x1:x2] for confidence, (x1, y1, x2, y2) in boxes]
        if as_rgb:
            faces = [cvtColor(face, code=COLOR_BGR2RGB) for face in faces]
        return faces