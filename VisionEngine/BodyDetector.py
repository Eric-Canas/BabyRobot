from cv2.dnn import blobFromImage, readNetFromCaffe
from Constants import *
import cv2
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
        self.network = readNetFromCaffe(prototxt=BODY_DETECTION_DEFINITION, caffeModel=BODY_DETECTION_WEIGHTS)
        self.input_size = INPUT_SIZE

    def predict(self, input):
        h, w = input.shape[:2]
        blob = blobFromImage(cv2.resize(input, self.input_size), scalefactor=INVERSE_STD,
                             size=self.input_size,
                             mean=MEAN, swapRB=False)
        self.network.setInput(blob)
        detections = self.network.forward()
        boxes = self.process_output(output=detections, h=h, w=w)
        return boxes


    def process_output(self, output, h, w):
        output = output[output[...,LABEL] == PERSON]
        output = output[output[...,CONFIDENCE] > MIN_CONFIDENCE]
        output = output[output[...,X2] <= 1.0]
        output = output[output[..., Y2] <= 1.0]
        return [(face[CONFIDENCE], (face[CONFIDENCE+1:] * np.array([w, h, w, h])).astype(np.int))
                for face in output]