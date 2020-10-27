from RecognitionPipeline import RecognitionPipeline


#CODES
GET_DISTANCE_TO_FACES_CODE, GET_DISTANCES_WITHOUT_IDENTITIES_CODE, SAVE_FILE_CODE = 0, 1, 2


class ServerPipeline:
    def __init__(self, self_socket, pipeline = None):
        self.socket = self_socket
        self.pipeline = pipeline if pipeline is not None else RecognitionPipeline()

    def execute_and_send_result(self, code, show=True):
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
