from Constants import NEW_PROPOSALS_DATA_DIR
from PiCamera.CameraCalculator.PiCameraV2Parameters import SENSOR_DIM_IN_PIXELS, FRAME_RATE
from picamera.array import PiRGBArray
from picamera import PiCamera as SuperPiCamera
from cv2 import waitKey, imwrite

import os
from random import randint

class PiCamera(SuperPiCamera):
    def __init__(self, resolution = SENSOR_DIM_IN_PIXELS, framerate_range = (FRAME_RATE, FRAME_RATE//3)):
        super().__init__(resolution=resolution, framerate_range=framerate_range)

    def capture_continuous(self):
        rawCapture = PiRGBArray(self, size=self.resolution)
        for image in super().capture_continuous(rawCapture, format="rgb", use_video_port=True):
            if waitKey(1) & 0xFF == ord("q"):
                break
            rawCapture.truncate(0)
            yield image.array

def capture_dataset(pipeline, save_at=NEW_PROPOSALS_DATA_DIR, persons = ("Albaby", "Eric")):
    [os.makedirs(os.path.join(save_at, person)) for person in persons if not os.path.exists(os.path.join(save_at, person))]
    with PiCamera() as camera:
        for image in camera.capture_continuous():
            faces = pipeline.get_faces_in_image(image=image)
            for name in persons:
                if name in faces:
                    imwrite(os.path.join(save_at, name, name+str(randint(0, 10000))+'.jpg'), faces[name])




