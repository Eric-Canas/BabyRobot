from cv2.dnn import blobFromImage, readNetFromCaffe
from Constants import *
from cv2 import resize
import numpy as np

LABEL = 1
PERSON = 15
CONFIDENCE = 2
X2, Y2 = -2, -1
MEAN = (123.68, 116.78, 103.94)
INVERSE_STD = 0.017
INPUT_SIZE = (304, 304)

class BodyDetector:
    def __init__(self):
        """
        Body detector, implemented through a pretrained model.
        """
        self.network = readNetFromCaffe(prototxt=BODY_DETECTION_DEFINITION, caffeModel=BODY_DETECTION_WEIGHTS)
        self.input_size = INPUT_SIZE

    def predict(self, input):
        """
        Prediction with the boxes that contain the body
        :param input: Numpy Image. Numpy image in the format BGR.
        :return: Boxes containing the detections.
        """
        h, w = input.shape[:2]
        blob = blobFromImage(resize(input, self.input_size), scalefactor=INVERSE_STD,
                             size=self.input_size,
                             mean=MEAN, swapRB=False)
        self.network.setInput(blob)
        detections = self.network.forward()
        boxes = self.process_output(output=detections, h=h, w=w)
        return boxes


    def process_output(self, output, h, w):
        """
        Process the raw output of the network for transforming it from the [0., 1.] space to the pixels space. It
        also erases the corrupted detections or those with a confidence lower than a MIN_CONFIDENCE threshold.
        :param output: Numpy. Last output of the network.
        :param h: Int. Original height of the input image.
        :param w: Int. Original width of the input image.
        :return: List of numpys in the format (x1, y1, x2, y2). Processed boxes.
        """
        output = output[output[...,LABEL] == PERSON]
        output = output[output[...,CONFIDENCE] > MIN_CONFIDENCE]
        output = output[output[...,X2] <= 1.0]
        output = output[output[..., Y2] <= 1.0]
        return [(face[CONFIDENCE], (face[CONFIDENCE+1:] * np.array([w, h, w, h])).astype(np.int))
                for face in output]