import io
import socket
import struct
from time import time
from Constants import DECIMALS
from PIL import Image
import numpy as np
import json
from RecognitionPipeline import sum_y_offset
from RobotController.ClientServer.ServerPipeline import GET_DISTANCE_TO_FACES_CODE, GET_DISTANCES_WITHOUT_IDENTITIES_CODE, SAVE_FILE_CODE
from RobotController.ClientServer.Socket import CODES

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
        self.socket = socket

    def get_distance_without_identities(self, image, y_offset=0.):
        self.socket.send_int_code(code=GET_DISTANCES_WITHOUT_IDENTITIES_CODE)
        self.socket.send_image(image=image)
        results = self.socket.receive_results()
        if len(results):
            results = sum_y_offset(distances=results, y_offset=y_offset)
        return results

    def get_distance_to_faces(self, image, y_offset=0., verbose=True):
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



