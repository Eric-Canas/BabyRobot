from cv2 import cvtColor, COLOR_BGR2RGB


class Detector:
    def __init__(self, model):
        """
        General class for executing a detector model.
        :param model: Usually BodyDetector or OpenCVFace Detector. Detector Model
        """
        self.model = model
        self.camera_calculator = ()

    def predict(self, image):
        """
        Return the prediction of the model for the given image.
        :param image: Numpy. Input image in format BGR.
        :return: List of numpy. Boxes detected.
        """
        return self.model.predict(image)

    def crop_boxes_content(self, image, boxes=None, as_rgb=False):
        """
        Crop a box inside the image for returning the sub-image that it contains.
        :param image: Numpy. Input image.
        :param boxes: List of Boxes in the format (x1, y1, x2, y2). Boxes to crop. If None, executes a prediction for
                                                                    obtaining them.
        :param as_rgb: Boolean. If True, swap the B and R channels.
        :return: List of numpy. Sub-images contained in the boxes.
        """
        if boxes is None:
            boxes = self.predict(image=image)
        faces = [image[y1:y2, x1:x2] for confidence, (x1, y1, x2, y2) in boxes]
        if as_rgb:
            faces = [cvtColor(face, code=COLOR_BGR2RGB) for face in faces]
        return faces