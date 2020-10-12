from cv2.dnn import blobFromImage, readNetFromCaffe
from Constants import FACE_DETECTOR_DIR, MIN_CONFIDENCE
from os.path import join
from cv2 import resize
import numpy as np

FACE_DETECTOR_DEFINITION = join(FACE_DETECTOR_DIR, 'deploy.prototxt')
FACE_DETECTOR_WEIGHTS = join(FACE_DETECTOR_DIR, 'weights.caffemodel')
INPUT_SIZE = (300,300)
TRAINING_MEAN = (104.0, 177.0, 123.0)

# Positions
CONFIDENCE = 2
X2, Y2 = -2, -1


class OpenCVFaceDetector:
    def __init__(self):
        self.network = readNetFromCaffe(prototxt=FACE_DETECTOR_DEFINITION, caffeModel=FACE_DETECTOR_WEIGHTS)
        self.input_size = INPUT_SIZE

    def predict(self, input):
        h, w = input.shape[:2]
        blob = blobFromImage(resize(input, self.input_size), scalefactor=1.0,
                             size=self.input_size,
                             mean=TRAINING_MEAN, swapRB=True)
        self.network.setInput(blob)
        detections = self.network.forward()
        boxes = self.process_output(output=detections, h=h, w=w)
        return boxes


    def process_output(self, output, h, w):
        output = output[output[...,CONFIDENCE] > MIN_CONFIDENCE]
        output = output[output[...,X2] <= 1.0]
        output = output[output[..., Y2] <= 1.0]
        return [(face[CONFIDENCE], (face[CONFIDENCE+1:] * np.array([w, h, w, h])).astype(np.int))
                for face in output]
