from PiCamera.CameraCalculator.CameraCalculator import CameraCalculator
from Printer import show_detections as show
from VisionEngine.Detector import Detector
from VisionEngine.DetectorModel import DetectorModel
from VisionEngine.Embedding import Embedding
from VisionEngine.MobileFaceNet import MobileFaceNet
from VisionEngine.Recognizer import Recognizer
from VisionEngine.BodyDetector import BodyDetector


class RecognitionPipeline:
    def __init__(self, face_detector = None, body_detector = None, embedding = None, classifier = None, camera_calculator = None):

        self.face_detector = face_detector if face_detector is not None else Detector(DetectorModel())
        self.body_detector = body_detector if body_detector is not None else Detector(BodyDetector())
        self.embedding = embedding if embedding is not None else Embedding(MobileFaceNet())
        self.classifier = classifier if classifier is not None else Recognizer(embedding=self.embedding) # binary_recognition='Albaby')
        self.camera_calculator = camera_calculator if camera_calculator is not None else CameraCalculator()

    def show_detections(self, image):
        face_boxes = self.face_detector.get_boxes(image=image)
        body_boxes = self.body_detector.get_boxes(image=image)
        confidences, names, distances = [], [], []
        if len(face_boxes):
            faces = self.face_detector.get_content(image=image, boxes=face_boxes)
            confidences.extend([confidence for confidence, _ in face_boxes])
            face_boxes = [box for _, box in face_boxes]
            embeddings = [self.embedding.predict(face) for face in faces]
            names.extend(self.classifier.predict(x=embeddings))

            distances.extend([self.camera_calculator.rectangleToRealWorldXY(rectangle=box, h=image.shape[0],
                                                                       w=image.shape[1]) for box in face_boxes])
        if len(body_boxes):
            #bodies = self.body_detector.get_content(image=image, boxes=body_boxes)
            confidences.extend([confidence for confidence, _ in body_boxes])
            body_boxes = [box for _, box in body_boxes]
            names.extend(['random_body' for box in body_boxes])
            distances.extend([self.camera_calculator.rectangleToRealWorldXY(rectangle=box, h=image.shape[0],
                                                                       w=image.shape[1]) for box in face_boxes])



        show(image=image, boxes=face_boxes+body_boxes, names=names, distances=distances, confidences=confidences)

    def get_faces_in_image(self, image):
        face_boxes = self.face_detector.get_boxes(image=image)
        faces = self.face_detector.get_content(image=image, boxes=face_boxes)
        embeddings = [self.embedding.predict(face) for face in faces]
        names = self.classifier.predict(x=embeddings)
        return {name : face for face, name in zip(faces, names)}






