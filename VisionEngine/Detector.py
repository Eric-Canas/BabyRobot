import cv2


class Detector:
    def __init__(self, model):
        self.model = model
        self.camera_calculator = ()

    def get_boxes(self, image):
        return self.model.predict(image)

    def get_content(self, image, boxes=None, as_rgb=False):
        if boxes is None:
            boxes = self.get_boxes(image=image)
        faces = [image[y1:y2, x1:x2] for confidence, (x1, y1, x2, y2) in boxes]
        if as_rgb:
            faces = [cv2.cvtColor(face, code=cv2.COLOR_BGR2RGB) for face in faces]
        return faces