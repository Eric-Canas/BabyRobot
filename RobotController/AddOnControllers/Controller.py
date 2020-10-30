from RobotController.AddOnControllers.MotorController import MotorController
from RobotController.AddOnControllers.BackUltraSoundController import BackUltraSoundController
from RobotController.AddOnControllers.CameraController import CameraController
from warnings import warn

class Controller:

    """
    SINGLETON
    This class include the controller and is in charge to send any signal to the Robot.
    As the DQN is an off-policy method and the unique access to the World is through exploration, this controller
    will keep the DQN containing the current Policy (The target model)
    """
    class __Controller:
        def __init__(self, motor_controller = None, back_ultrasound_controller = None, camera_controller = None):
            self.motor_controller = motor_controller if motor_controller is not None else MotorController()
            self.back_ultrasound_controller = back_ultrasound_controller if back_ultrasound_controller is not None else BackUltraSoundController()
            self.camera_controller = camera_controller if camera_controller is not None else CameraController()
            self.capture_continuous = self.camera_controller.capture_continuous

    instance = None
    def __init__(self, motor_controller = None, back_ultrasound_controller = None):
        if Controller.instance is None:
            Controller.instance = Controller.__Controller(motor_controller=motor_controller, back_ultrasound_controller=back_ultrasound_controller)
        else:
            warn("Trying to reinstantiate a the Singleton class controller. Instantation skipped")

    def __getattr__(self, item):
        return getattr(self.instance, item)

    def __setattr__(self, key, value):
        return setattr(self.instance, key, value)

    def idle(self, time = None):
        self.motor_controller.idle(time=time)

    def move_forward(self, time = None):
        self.motor_controller.move_straight(front = True, time = time)

    def move_back(self, time = None):
        self.motor_controller.move_straight(front = False, time = time)

    def rotate_clockwise(self, time = None):
        self.motor_controller.rotate(clockwise = True, time = time)

    def rotate_counterclockwise(self, time = None):
        self.motor_controller.rotate(clockwise = False, time = time)

    def go_right_front(self, time = None):
        self.motor_controller.turn(right = True, front=True, time = time, )

    def go_left_front(self, time = None):
        self.motor_controller.turn(right = False, front=True, time = time)

    def go_right_back(self, time = None):
        self.motor_controller.turn(right = True, front=False, time = time)

    def go_left_back(self, time = None):
        self.motor_controller.turn(right = False, front=False, time = time)

    def get_back_distance(self, back_distance_offset = 0.):
        return self.back_ultrasound_controller.get_distance()-back_distance_offset

    def capture_image(self):
        return self.camera_controller.capture()