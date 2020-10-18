from Constants import NEW_PROPOSALS_DATA_DIR
from RobotController.PiCamera.CameraCalculator.PiCameraV2Parameters import PREFERRED_RESOLUTION, FRAME_RATE

import cv2
import os
from random import randint
from warnings import warn

try:
    from picamera.array import PiRGBArray
    from picamera import PiCamera as SuperPiCamera

    class PiCamera(SuperPiCamera):
        class __PiCamera(SuperPiCamera):
            def __init__(self, resolution=PREFERRED_RESOLUTION, framerate_range=(FRAME_RATE, FRAME_RATE // 3)):
                super().__init__(resolution=resolution, framerate_range=framerate_range)
                self.raw_capture = PiRGBArray(self, size=self.resolution)

        instance = None

        def __init__(self, resolution=PREFERRED_RESOLUTION, framerate_range=(FRAME_RATE, FRAME_RATE // 3)):
            if PiCamera.instance is None:
                PiCamera.instance = PiCamera.__PiCamera(resolution=resolution, framerate_range=framerate_range)
            else:
                warn("Trying to reinstantiate a the Singleton class controller. Instantation skipped")

        def __getattr__(self, item):
            return getattr(self.instance, item)

        def __setattr__(self, key, value):
            return setattr(self.instance, key, value)

        def capture_continuous(self):
            self.raw_capture.truncate(0)
            for image in super().capture_continuous(self.raw_capture, format="rgb", use_video_port=True):
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                self.raw_capture.truncate(0)
                yield image.array

        def capture(self):
            self.raw_capture.truncate(0)
            return super().capture(self.raw_capture, 'rgb')
except:
    warn("picamera module not found. PiCamera will be a Mock Object")
    from VisionEngine.Dataset import read_image
    from Constants import FULL_PHOTOS_DATA_DIR
    from random import choice
    class PiCamera():
        def __init__(self, resolution = PREFERRED_RESOLUTION, framerate_range = (FRAME_RATE, FRAME_RATE//3)):
            self.resolution = resolution
            self.framerate_range = framerate_range
        def capture_continuous(self):
            while True:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                yield self.capture()

        def capture(self):
            person_dir = os.path.join(FULL_PHOTOS_DATA_DIR, choice(os.listdir(FULL_PHOTOS_DATA_DIR)))
            image = read_image(height=self.resolution[0], width= self.resolution[1],person_dir=person_dir, image_name=choice(os.listdir(person_dir)))
            return image


def capture_dataset(pipeline, save_at=NEW_PROPOSALS_DATA_DIR, persons = ("Albaby", "Eric")):
    [os.makedirs(os.path.join(save_at, person)) for person in persons if not os.path.exists(os.path.join(save_at, person))]
    with PiCamera() as camera:
        for image in camera.capture_continuous():
            faces = pipeline.get_faces_in_image(image=image)
            for name in persons:
                if name in faces:
					image = cv2.cvtColor(faces[name], code=cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(save_at, name, name+str(randint(0, 10000))+'.jpg'), image)




