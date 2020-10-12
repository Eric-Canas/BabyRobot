from RobotController.PiCamera.PiCamera import PiCamera

class CameraController:
    def __init__(self, camera = None):
        self.camera = camera if camera is not None else PiCamera()
        self.capture_continuous = self.camera.capture_continuous

    def capture(self):
        return self.camera.capture()

