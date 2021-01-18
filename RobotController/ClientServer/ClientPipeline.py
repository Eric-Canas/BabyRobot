
from time import time
from Constants import DECIMALS
from RecognitionPipeline import sum_y_offset
from RobotController.ClientServer.ServerPipeline import GET_DISTANCE_TO_FACES_CODE, GET_DISTANCES_WITHOUT_IDENTITIES_CODE

try:
    # If we are in Raspberry, it will always be the client
    import RPi.GPIO as GPIO
    CLIENT = True
except:
    # If we are in Computer, it will always be the server
    CLIENT = False

#RASPBERRY_HOSTNAME = 'raspberrypi'
SERVER_HOSTNAME = 'HaruPC'
TCP_PORT = 2020
STRUCT_FORMAT = '<L'

class ClientPipeline:
    def __init__(self, socket):
        """
        Client socket that executes the Computer Vision pipeline in the Server. Sends the image to the server
        and receives the prediction
        :param socket: Socket. Client socket already connected with the server.
        """
        self.socket = socket

    def get_distance_without_identities(self, image, y_offset=0.):
        """
        Executes the get_distance_without_identities function of the Computer Vision Pipeline but in the Server.
        :param image: Numpy. Image to analyze
        :param y_offset: Float. Offset to apply to the Y distance returned (usually the objective distance
                                to maintain with the target user)
        :return: Results of the get_distance_without_identity function of the Computer Vision Pipeline
        """
        self.socket.send_int_code(code=GET_DISTANCES_WITHOUT_IDENTITIES_CODE)
        self.socket.send_image(image=image)
        results = self.socket.receive_results()
        if len(results):
            results = sum_y_offset(distances=results, y_offset=y_offset)
        return results

    def get_distance_to_faces(self, image, y_offset=0., verbose=True):
        """
        Executes the get_distance_to_faces function of the Computer Vision Pipeline but in the Server.
        :param image: Numpy. Image to analyze
        :param y_offset: Float. Offset to apply to the Y distance returned (usually the objective distance
                                to maintain with the target user)
        :param verbose: Boolean. If True verbose the time performance.
        :return: Results of the get_distance_to_faces function of the Computer Vision Pipeline
        """
        self.socket.send_int_code(code=GET_DISTANCE_TO_FACES_CODE)
        if verbose: start_time = time()
        self.socket.send_image(image=image)
        if verbose:
            print(
                "Image sent to the server in {seconds} s:".format(seconds=round(time() - start_time, ndigits=DECIMALS)))
            start_time = time()
        results = self.socket.receive_results()
        if len(results):
            results = sum_y_offset(distances=results, y_offset=y_offset)
        if verbose: print(
            "Results received in {seconds} s:".format(seconds=round(time() - start_time, ndigits=DECIMALS)))
        return results



