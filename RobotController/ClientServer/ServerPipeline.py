from RecognitionPipeline import RecognitionPipeline


#CODES
GET_DISTANCE_TO_FACES_CODE, GET_DISTANCES_WITHOUT_IDENTITIES_CODE, SAVE_FILE_CODE = 0, 1, 2


class ServerPipeline:
    def __init__(self, self_socket, pipeline = None):
        """
        Receives codes identifying functions of the RecognitionPipeline and images.
        Execute them and return the result to the client.
        :param self_socket: Socket. Server socket already connected with the client.
        :param pipeline: RecognitionPipeline. Computer Vision pipeline to execute
        """
        self.socket = self_socket
        self.pipeline = pipeline if pipeline is not None else RecognitionPipeline()

    def execute_and_send_result(self, code, show=True):
        """
        Receives codes identifying functions of the RecognitionPipeline and images.
        Execute them and return the result to the client.
        :param code: Int. Codes identifying one of the functions of the pipeline that can be computed in the
                          Client/Server mode. 0 for get_distance_without_identities and 1 for get_distance_to_faces.
        :param show: Boolean. If True, show the image received with the detections in the screen.
        """
        image = self.socket.receive_image()
        if code == GET_DISTANCES_WITHOUT_IDENTITIES_CODE:
            result = self.pipeline.get_distance_without_identities(image=image, y_offset=0.)
        elif code == GET_DISTANCE_TO_FACES_CODE:
            result =  self.pipeline.get_distance_to_faces(image=image, y_offset=0.)
        else:
            raise NotImplementedError()
        self.socket.send_results(results=result)
        if show:
            self.pipeline.show_detections(image=image)
