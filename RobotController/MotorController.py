import RPi.GPIO as GPIO
from time import sleep

MOTOR1_CLOCKWISE = 27
MOTOR1_COUNTERCLOCKWISE = 17

MOTOR2_CLOCKWISE = 24
MOTOR2_COUNTERCLOCKWISE = 23

MOVEMENT_TIME = 1

class MotorController:
    def __init__(self, motor1_clockwise = MOTOR1_CLOCKWISE, motor1_counterclockwise = MOTOR1_COUNTERCLOCKWISE,
                 motor2_clockwise = MOTOR2_CLOCKWISE, motor2_counterclockwise = MOTOR2_COUNTERCLOCKWISE,
                 movement_time = MOVEMENT_TIME):
        self.movement_time = movement_time

        self.motor_left_clockwise = motor1_clockwise
        self.motor_left_counterclockwise = motor1_counterclockwise
        self.motor_right_clockwise = motor2_clockwise
        self.motor_right_clockwise = motor2_counterclockwise

        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.motor_left_clockwise, GPIO.OUT)
        GPIO.setup(self.motor_left_counterclockwise, GPIO.OUT)

        GPIO.setup(self.motor_right_clockwise, GPIO.OUT)
        GPIO.setup(self.motor_right_clockwise, GPIO.OUT)

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self):
        GPIO.cleanup()

    def stop(self):
        GPIO.output(self.motor_left_clockwise, False)
        GPIO.output(self.motor_left_counterclockwise, False)
        GPIO.output(self.motor_right_clockwise, False)
        GPIO.output(self.motor_right_clockwise, False)

    def move_straight(self, front=True, time=None):
        GPIO.output(self.motor_left_clockwise, front)
        GPIO.output(self.motor_left_counterclockwise, not front)
        GPIO.output(self.motor_right_clockwise, front)
        GPIO.output(self.motor_right_clockwise, not front)

        sleep(time if time is not None else self.movement_time)
        self.stop()

    def rotate(self, clockwise=True, time=None):
        GPIO.output(self.motor_left_clockwise, clockwise)
        GPIO.output(self.motor_left_counterclockwise, not clockwise)
        GPIO.output(self.motor_right_clockwise, clockwise)
        GPIO.output(self.motor_right_clockwise, not clockwise)
        sleep(time if time is not None else self.movement_time)
        self.stop()


