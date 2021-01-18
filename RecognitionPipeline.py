from RobotController.PiCamera.CameraCalculator.CameraCalculator import CameraCalculator
from VisionEngine.Recognizer import Recognizer
from Printer import show_detections as show
from VisionEngine.Detector import Detector
from VisionEngine.OpenCVFaceDetector import OpenCVFaceDetector
from VisionEngine.Embedding import Embedding
from VisionEngine.MobileFaceNet import MobileFaceNet
from VisionEngine.BodyDetector import BodyDetector
from time import time
from cv2 import cvtColor, COLOR_BGR2RGB
from Constants import CRAWLING_BODY_HEIGHT_IN_CM

class RecognitionPipeline:
    def __init__(self, face_detector = None, body_detector = None, embedding = None, classifier = None,
                 camera_calculator = None, show_bodies = False):
        """
        Whole recognition pipeline that will receive an image as input an will detect and identify the faces in it.
        It will detect the bodies if no faces appear in the image.
        :param face_detector: Detector. Face detector to use.
        :param body_detector: Detector. Body detector to use as B-Plan.
        :param embedding: Emdedding. Embedder to use for embedding the faces detected.
        :param classifier: Recognizer. Recognition pipeline to use for identifying the detected faces.
        :param camera_calculator: CameraCalculator. CameraCalculator to use for measuring distances
        :param show_bodies: Boolean. If True, search for bodies as B-Plan if no faces are detected.
        """
        self.face_detector = face_detector if face_detector is not None else Detector(OpenCVFaceDetector())
        self.show_bodies = show_bodies
        self.body_detector = body_detector if body_detector is not None else Detector(BodyDetector())
        self.embedding = embedding if embedding is not None else Embedding(MobileFaceNet())
        self.classifier = classifier if classifier is not None else Recognizer(embedding=self.embedding)
        self.camera_calculator = camera_calculator if camera_calculator is not None else CameraCalculator()

    def show_recognitions(self, image, verbose = False, return_faces = False):
        """
        Detects the faces and bodies in an image and plots all the recognitions in a window.
        :param image: Numpy. Input Image in format BGR.
        :param verbose: Boolean. If true, verboses the inference time.
        :param return_faces: Boolean. If True, return the detected faces as input
        :return: {String : Numpy Image}. If return_faces is true, returns a dictionary with the faces detected.
        """
        if verbose:
            start_time = time()
        boxes = self.face_detector.predict(image=image)
        if verbose:
            face_detection_time = time() - start_time
        confidences, names, distances = [], [], []
        if len(boxes):
            faces = self.face_detector.crop_boxes_content(image=image, boxes=boxes)
            confidences.extend([confidence for confidence, _ in boxes])
            boxes = [box for _, box in boxes]

            if verbose:
                embedding_start_time = time()
            embeddings = [self.embedding.predict(face) for face in faces]
        
            if verbose:
                embedding_time = time() - embedding_start_time
            if verbose:
                recognition_start_time = time()
            names.extend(self.classifier.predict(x=embeddings))
            if verbose:
                recognition_time = time() - recognition_start_time
            distances.extend([self.camera_calculator.rectangleToRealWorldXY(rectangle=box, h=image.shape[0],
                                                                       w=image.shape[1]) for box in boxes])
        if self.show_bodies:
            if verbose:
                bodies_start_time = time()
            body_boxes = self.body_detector.predict(image=cvtColor(image, code=COLOR_BGR2RGB))
            if verbose:
                body_time = time() - bodies_start_time
            if len(body_boxes):
                #bodies = self.body_detector.get_content(image=image, boxes=body_boxes)
                confidences.extend([confidence for confidence, _ in body_boxes])
                body_boxes = [box for _, box in body_boxes]
                names.extend(['body' for box in body_boxes])
                distances.extend([self.camera_calculator.rectangleToRealWorldXY(rectangle=box, h=image.shape[0],
                                                                           w=image.shape[1]) for box in boxes])
                boxes = boxes + body_boxes

        if verbose:
            showing_start_time = time()
        show(image=image, boxes=boxes, names=names, distances=distances, confidences=confidences)

        if verbose and len(boxes):
            showing_time = time() - showing_start_time
            total_time = time()-start_time
            if not self.show_bodies or len(boxes) > len(body_boxes):
                txt = 'Total prediction time: {total_time} s\n' \
                      '\t Face Detection: {face_time} s ({face_p})%\n' \
                      '\t Face Embedding: {embedding_time} s ({embedd_p})%\n' \
                      '\t Face Recognition: {recognition_time} s ({recognition_p})%\n'\
                    .format(total_time = round(total_time, ndigits=2),
                            face_time=round(face_detection_time, ndigits=2), face_p=round((face_detection_time/total_time)*100, ndigits=2),
                            embedding_time=round(embedding_time, ndigits=2), embedd_p=round((embedding_time/total_time)*100, ndigits=2),
                            recognition_time=round(recognition_time, ndigits=2), recognition_p=round((recognition_time/total_time)*100, ndigits=2))
            if self.show_bodies:
                txt += '\t Body Detection: {detection_time} s ({detection_p})%\n'\
                    .format(detection_time=round(body_time, ndigits=2), detection_p=round((body_time/total_time)*100, ndigits=2))
            txt += '\t Plotting time: {plot_time} s ({plotting_p})%'\
                .format(plot_time=round(showing_time, ndigits=2), plotting_p=round((showing_time/total_time)*100, ndigits=2))
            print(txt)
        if return_faces:
            if len(faces):
                names = names[:len(faces)]
                return {name : face for face, name in zip(faces, names)}
        else:
            return {}

    def show_detections(self, image):
        """
        Plot the detections without any associated identification.
        :param image: Numpy. Input Image in format BGR.
        """
        boxes = self.face_detector.predict(image=image)
        if len(boxes) == 0:
            boxes = self.body_detector.predict(image=image)
        confidences = [conf for conf, _ in boxes]
        boxes = [box for _, box in boxes]
        distances = [self.camera_calculator.rectangleToRealWorldXY(rectangle=box, h=image.shape[0],
                                                       w=image.shape[1]) for box in boxes]

        show(image=image, boxes=boxes, names=None, distances=distances, confidences=confidences)

    def get_faces_in_image(self, image):
        """
        Return the sub-images that contains a face in 'image' including the name associated to each face.
        :param image: Numpy. Input Image in format BGR.
        :return: {Name : Numpy Image}. Dictionary with the sub-image of each detected face with its identification.
        """
        face_boxes = self.face_detector.predict(image=image)
        if len(face_boxes) > 0:
            faces = self.face_detector.crop_boxes_content(image=image, boxes=face_boxes)
            embeddings = [self.embedding.predict(face) for face in faces]
            names = self.classifier.predict(x=embeddings)
            return {name : face for face, name in zip(faces, names)}
        else:
            return {}

    def get_distance_to_faces(self, image, y_offset = 0.):
        """
        Measure the distances to the faces that appears in an image
        :param image: Numpy. Input Image in format BGR.
        :param y_offset: Float. Y offset to apply to the detection. Usually the preferred distance to maintain with
                                the target, so distance will indicate the deviation to it.
        :return: {Name : Float}. Dictionary with the distances associated to each person in the image.
        """
        face_boxes = self.face_detector.predict(image=image)
        if len(face_boxes):
            faces = self.face_detector.crop_boxes_content(image=image, boxes=face_boxes)
            face_boxes = [box for _, box in face_boxes]
            embeddings = [self.embedding.predict(face) for face in faces]
            names = self.classifier.predict(x=embeddings)

            distances = [self.camera_calculator.rectangleToRealWorldXY(rectangle=box, h=image.shape[0],
                                                                            w=image.shape[1]) for box in face_boxes]
            distances = sum_y_offset(distances=distances, y_offset=y_offset)
        else:
            distances, names = [], []

        return {name : distances for distance, name in zip(distances, names)}

    def get_distance_without_identities(self, image, y_offset=0., detect_body_if_face_not_found = True):
        """
        Measure the distances to the faces or bodies that appears in an image
        :param image: Numpy. Input Image in format BGR.
        :param y_offset: Float. Y offset to apply to the detection. Usually the preferred distance to maintain with
                                the target, so distance will indicate the deviation to it.
        :param detect_body_if_face_not_found: Boolean. If True, try to detect bodies if no faces appear in the image.
        :return: List of Float. List with the distances associated to any detected face or body
                                (without any identification).
        """
        boxes = self.face_detector.predict(image=image)
        element_height = None
        if detect_body_if_face_not_found and len(boxes) == 0:
            boxes = self.body_detector.predict(image=image)
            element_height = CRAWLING_BODY_HEIGHT_IN_CM
        if len(boxes):
            boxes = [box for conf, box in boxes]
            distances = self.camera_calculator.rectangleToRealWorldXY(rectangle=boxes[0], h=image.shape[0],
                                                                      w=image.shape[1], element_height_in_cm=element_height)
            return sum_y_offset(distances=distances, y_offset=y_offset)
        else:
            return ()

def sum_y_offset(distances, y_offset=0.):
    """
    For every tuple in a list of (x,y) positions, adds an offset to y.
    :param distances: List of tuples (Int, Int). List with the (x, y) positions to which sum the offset.
    :param y_offset: Float. Offset to add.
    :return: List of tuples (Int, Int). List of the (x, y) with the offset added to Y.
    """
    if type(distances) is dict:
        raise NotImplementedError()
    elif type(distances[0]) not in [list, tuple]:
        return (distances[0], distances[1] - y_offset)
    else:
        return [(x, y-y_offset) for x,y in distances]


