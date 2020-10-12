from time import sleep
from warnings import warn

MOTOR1_CLOCKWISE = 27
MOTOR1_COUNTERCLOCKWISE = 17

MOTOR2_CLOCKWISE = 24
MOTOR2_COUNTERCLOCKWISE = 23

MOVEMENT_TIME = 1
try:
    import RPi.GPIO as GPIO
    class MotorController:
        class __MotorController:
            def __init__(self, motor1_clockwise_pin = MOTOR1_CLOCKWISE, motor1_counterclockwise_pin = MOTOR1_COUNTERCLOCKWISE,
                     motor2_clockwise_pin = MOTOR2_CLOCKWISE, motor2_counterclockwise_pin = MOTOR2_COUNTERCLOCKWISE,
                     default_movement_time = MOVEMENT_TIME):

                self.default_movement_time = default_movement_time

                self.motor_left_clockwise = motor1_clockwise_pin
                self.motor_left_counterclockwise = motor1_counterclockwise_pin
                self.motor_right_clockwise = motor2_clockwise_pin
                self.motor_right_clockwise = motor2_counterclockwise_pin

                GPIO.setmode(GPIO.BCM)

                GPIO.setup(self.motor_left_clockwise, GPIO.OUT)
                GPIO.setup(self.motor_left_counterclockwise, GPIO.OUT)

                GPIO.setup(self.motor_right_clockwise, GPIO.OUT)
                GPIO.setup(self.motor_right_clockwise, GPIO.OUT)

        instance = None

        def __init__(self, motor1_clockwise_pin = MOTOR1_CLOCKWISE, motor1_counterclockwise_pin = MOTOR1_COUNTERCLOCKWISE,
                     motor2_clockwise_pin = MOTOR2_CLOCKWISE, motor2_counterclockwise_pin = MOTOR2_COUNTERCLOCKWISE,
                     default_movement_time = MOVEMENT_TIME):
            if MotorController.instance is None:
                MotorController.instance = MotorController.__MotorController(motor1_clockwise_pin=motor1_clockwise_pin,
                                                                             motor1_counterclockwise_pin=motor1_counterclockwise_pin,
                                                                             motor2_clockwise_pin=motor2_clockwise_pin,
                                                                             motor2_counterclockwise_pin=motor2_counterclockwise_pin,
                                                                             default_movement_time=default_movement_time)
            else:
                warn("Trying to re-instantiate a the Singleton class controller. Instantiation skipped")

        def __getattr__(self, item):
            return getattr(self.instance, item)

        def __setattr__(self, key, value):
            return setattr(self.instance, key, value)

        def __exit__(self, exc_type, exc_value, traceback):
            self.cleanup()
            MotorController.instance = None

        def cleanup(self):
            GPIO.cleanup()

        def stop(self):
            GPIO.output(self.motor_left_clockwise, False)
            GPIO.output(self.motor_left_counterclockwise, False)
            GPIO.output(self.motor_right_clockwise, False)
            GPIO.output(self.motor_right_counterclockwise, False)

        def move_straight(self, front=True, time=None):
            GPIO.output(self.motor_left_clockwise, front)
            GPIO.output(self.motor_left_counterclockwise, not front)
            GPIO.output(self.motor_right_clockwise, front)
            GPIO.output(self.motor_right_counterclockwise, not front)

            sleep(time if time is not None else self.default_movement_time)
            self.stop()

        def rotate(self, clockwise=True, time=None):
            GPIO.output(self.motor_left_clockwise, clockwise)
            GPIO.output(self.motor_left_counterclockwise, not clockwise)
            GPIO.output(self.motor_right_clockwise, clockwise)
            GPIO.output(self.motor_right_counterclockwise, not clockwise)
            sleep(time if time is not None else self.default_movement_time)
            self.stop()
except:
    warn("GPIO module not found. MotorController will be a Mock Object")
    """
    Do a Mock Object for testing without the Raspberry
    """
    class MotorController:
        def __init__(self, motor1_clockwise_pin=MOTOR1_CLOCKWISE, motor1_counterclockwise_pin=MOTOR1_COUNTERCLOCKWISE,
                     motor2_clockwise_pin=MOTOR2_CLOCKWISE, motor2_counterclockwise_pin=MOTOR2_COUNTERCLOCKWISE,
                     default_movement_time=MOVEMENT_TIME):
            pass

        def move_straight(self, front=True, time=None):
            pass

        def rotate(self, clockwise=True, time=None):
            pass