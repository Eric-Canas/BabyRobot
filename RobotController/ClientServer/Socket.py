import io
import socket
import struct
from time import time
from Constants import DECIMALS
from PIL import Image
import numpy as np
import json
from RobotController.ClientServer.ServerPipeline import SAVE_FILE_CODE, GET_DISTANCE_TO_FACES_CODE, \
    GET_DISTANCES_WITHOUT_IDENTITIES_CODE
from os import SEEK_END
import os
import pickle

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
STR_ENCODING = 'utf-8'
PKL_EXTENSION = '.pkl'
class Socket:
    def __init__(self, client=CLIENT, server_hostname = SERVER_HOSTNAME, tcp_port=TCP_PORT, ip=None, verbose=True):
        self.client = client
        self.tcp_port = tcp_port
        self.socket = socket.socket(socket.AF_INET , socket.SOCK_STREAM)

        if ip is not None:
            self.server_ip = ip
        else:
            self.server_ip = socket.gethostbyname(server_hostname)

        if self.client:
            self.server_hostname = server_hostname
            if verbose: print("Trying to connect to the server located at {name}:{port} ({ip})...".format(name=self.server_hostname, port=self.tcp_port, ip=self.server_ip))
            self.socket.connect((self.server_ip, self.tcp_port))
            if verbose: print("Connected.")
            self.connection = self.socket.makefile(mode='wb')
        else:
            self.server_hostname = socket.gethostname()
            if verbose: print("Opening a TCP server at {name}:{port} ({ip})...".format(name=self.server_hostname, port=self.tcp_port, ip=self.server_ip))
            self.socket.bind((self.server_ip, self.tcp_port))
            if verbose: print("Conexion opened.")
            self.socket.listen(True)
            self.client_socket = self.socket.accept()[0]
            self.connection = self.client_socket.makefile(mode='rb')
        self.verbose = verbose



    def send_image(self, image):
        if self.verbose: start_time = time()
        pil_img = Image.fromarray(image)
        with io.BytesIO() as stream:
            pil_img.save(stream, "JPEG")
            # Writes the header
            image_len = stream.tell()
            self.connection.write(struct.pack(STRUCT_FORMAT, image_len))
            self.connection.flush()
            stream.seek(0)
            # Writes the stream
            self.connection.write(stream.read())
            self.connection.flush()

        if self.verbose: print("Image sent to server {name}:{port} ({ip}) in {seconds} ({kbsize} Kb)"
                               .format(name=self.server_hostname, port=self.tcp_port, ip=self.server_ip,
                                       seconds=round(time()-start_time, ndigits=DECIMALS), kbsize=round(image_len/1024, ndigits=DECIMALS)))

    def send_results(self, results):
        results = bytes(json.dumps(results), encoding=STR_ENCODING)
        self.client_socket.send(struct.pack(STRUCT_FORMAT, len(results)))
        self.client_socket.sendall(results)
        if self.verbose: print("Results sent back to client ({length} b)".format(length=len(results)))

    def receive_results(self):
        # Reads the header for obtaining the image size (connection-read blocks the execution until is received)
        with io.StringIO() as stream:
            if self.verbose: start_time = time()
            results_len = struct.unpack(STRUCT_FORMAT, self.socket.recv(struct.calcsize(STRUCT_FORMAT)))[0]
            # Reads the image (connection-read blocks the execution until is received)
            stream.write(self.socket.recv(results_len).decode(STR_ENCODING))
            stream.seek(0)
            results = json.load(stream)
            if self.verbose: print("Results received in {seconds} s ({bsize} b)".format(seconds=round(time()-start_time, ndigits=DECIMALS), bsize=stream.tell()))
        return results

    def send_int_code(self, code):
        code_in_bytes = bytes(str(code), encoding=STR_ENCODING)
        self.socket.send(code_in_bytes)
        if self.verbose: print("Sent int code for executing: {meth}.".format(meth=CODES[code]))

    def receive_int_code(self):
        code = int(self.client_socket.recv(1).decode(STR_ENCODING))
        if self.verbose: print("Received int code for executing: {meth}.".format(meth=CODES[code]))
        return code

    def receive_image(self):
        if self.verbose: start_time = time()
        # Reads the header for obtaining the image size (connection-read blocks the execution until is received)
        image_len = struct.unpack(STRUCT_FORMAT, self.connection.read(struct.calcsize(STRUCT_FORMAT)))[0]
        with io.BytesIO() as stream:
            # Reads the image (connection-read blocks the execution until is received)
            stream.write(self.connection.read(image_len))
            stream.seek(0)
            received_image = np.array(Image.open(stream))
            if self.verbose: print("Image received in {seconds} s ({kbsize} kb)".format(seconds=round(time()-start_time, ndigits=DECIMALS), kbsize=round(image_len/1024, ndigits=DECIMALS)))
        return received_image

    def send_file(self, file_path, verbose=True):
        self.send_int_code(code=SAVE_FILE_CODE)
        with open(file_path, mode='rb') as stream:
            # Send file name length
            if verbose:
                start_time = time()
                print("File {filepath} will be sent...".format(filepath=file_path))
            file_path_as_bytes = bytes(file_path, encoding=STR_ENCODING)
            self.connection.write(struct.pack(STRUCT_FORMAT, len(file_path_as_bytes)))
            self.connection.flush()
            #Send file name
            self.connection.write(file_path_as_bytes)
            self.connection.flush()
            if file_path.endswith(PKL_EXTENSION):
                data = pickle.dumps(pickle.load(stream))
                self.socket.send(struct.pack(STRUCT_FORMAT, len(data)))
                self.socket.sendall(data)
            else:
                stream.seek(0, SEEK_END)
                self.connection.write(struct.pack(STRUCT_FORMAT, stream.tell()))
                self.connection.flush()
                stream.seek(0)
                # Writes the stream
                self.connection.write(stream.read())
                self.connection.flush()
                if verbose: print("File sent in {seconds} s ({kbsize} Kb)".format(seconds=round(time()-start_time, ndigits=DECIMALS), kbsize=round(stream.tell()/1024, ndigits=DECIMALS)))

    def receive_file(self, save_it = True, verbose=True):
        file_path_length = struct.unpack(STRUCT_FORMAT, self.connection.read(struct.calcsize(STRUCT_FORMAT)))[0]
        file_path = self.connection.read(file_path_length).decode(STR_ENCODING)
        if verbose:
            print("Preparing for receive {f}".format(f=file_path))
            start_time = time()

        if not file_path.endswith(PKL_EXTENSION):
            file_len = struct.unpack(STRUCT_FORMAT, self.connection.read(struct.calcsize(STRUCT_FORMAT)))[0]
            data = self.connection.read(file_len)
            stream = io.BytesIO(data)
            stream.seek(0)
        else:
            file_len = struct.unpack(STRUCT_FORMAT, self.client_socket.recv(struct.calcsize(STRUCT_FORMAT)))[0]
            data = self.client_socket.recv(file_len)
            data = pickle.loads(data)
        if verbose: print("File {filepath} received in {seconds} s ({kbsize} Kb)".format(filepath=file_path,
                                                                                         seconds=round(time() - start_time, ndigits=DECIMALS),
                                                                                         kbsize=round(file_len / 1024, ndigits=DECIMALS)))
        if save_it:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(file_path, mode='wb') as file:
                if file_path.endswith(PKL_EXTENSION):
                    pickle.dump(data, file=file)
                else:
                    file.write(stream.read())
                    stream.close()
            if verbose: print("File saved")
            return None
        else:
            return {file_path : (stream if not file_path.endswith(PKL_EXTENSION) else io.StringIO(data))}


    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Everythig closed")
        self.connection.close()
        self.socket.close()


CODES = {GET_DISTANCE_TO_FACES_CODE : 'get_distances_to_faces()',
         GET_DISTANCES_WITHOUT_IDENTITIES_CODE : 'get_distances_without_identities()',
         SAVE_FILE_CODE : 'save_file()'}