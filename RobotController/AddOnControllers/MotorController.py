"""
If the code is executed in a Raspberry, define MotorController as the communicator with the L298N.
Elsewhere instantiate a Mock Object for debugging in the PC.
"""

from time import sleep
from warnings import warn
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from RobotController.RLConstants import MOVEMENT_TIME, DEFAULT_MOVEMENT_MODE, SYNC_MODE, ASYNC_MODE, HALF_MODE, ONE_THIRD_MODE

MOTOR1_CLOCKWISE = 17
MOTOR1_COUNTERCLOCKWISE = 27

MOTOR2_CLOCKWISE = 23
MOTOR2_COUNTERCLOCKWISE = 24



try:
    import RPi.GPIO as GPIO
    """
    Define MotorController as a Singleton class, for avoiding multiple controllers pointing to the same physical L298N
    """
    class MotorController:
        class __MotorController:
            def __init__(self, motor1_clockwise_pin = MOTOR1_CLOCKWISE, motor1_counterclockwise_pin = MOTOR1_COUNTERCLOCKWISE,
                         motor2_clockwise_pin = MOTOR2_CLOCKWISE, motor2_counterclockwise_pin = MOTOR2_COUNTERCLOCKWISE,
                         default_movement_time =MOVEMENT_TIME, movement_mode = DEFAULT_MOVEMENT_MODE):

                self.default_movement_time = default_movement_time

                self.motor_left_clockwise = motor1_clockwise_pin
                self.motor_left_counterclockwise = motor1_counterclockwise_pin
                self.motor_right_clockwise = motor2_clockwise_pin
                self.motor_right_counterclockwise = motor2_counterclockwise_pin

                GPIO.setmode(GPIO.BCM)

                GPIO.setup(self.motor_left_clockwise, GPIO.OUT)
                GPIO.setup(self.motor_left_counterclockwise, GPIO.OUT)

                GPIO.setup(self.motor_right_clockwise, GPIO.OUT)
                GPIO.setup(self.motor_right_counterclockwise, GPIO.OUT)
                self.movement_mode = movement_mode
                if self.movement_mode != SYNC_MODE:
                    self.scheduler = BackgroundScheduler(daemon=True)
                    self.scheduler.start()



        instance = None

        def __init__(self, motor1_clockwise_pin = MOTOR1_CLOCKWISE, motor1_counterclockwise_pin = MOTOR1_COUNTERCLOCKWISE,
                     motor2_clockwise_pin = MOTOR2_CLOCKWISE, motor2_counterclockwise_pin = MOTOR2_COUNTERCLOCKWISE,
                     default_movement_time =MOVEMENT_TIME, movement_mode = DEFAULT_MOVEMENT_MODE):
            """
            Instantiates the motor controller or returns the already instantiated one (Singleton).
            :param motor1_clockwise_pin: Int. GPIO pin for Motor 1 Clockwise.
            :param motor1_counterclockwise_pin: Int. GPIO pin for Motor 1 Counterclockwise.
            :param motor2_clockwise_pin: Int. GPIO pin for Motor 2 Clockwise.
            :param motor2_counterclockwise_pin: Int. GPIO pin for Motor 2 Clockwise.
            :param default_movement_time: Float. Default movement for every action (not applied when function receives
                                                 a time parameter different than None).
            :param movement_mode: String. Movement mode. One of: 'sync', 'async', '50-50', '33-66' (Default is 'sync').
            """
            if MotorController.instance is None:
                MotorController.instance = MotorController.__MotorController(motor1_clockwise_pin=motor1_clockwise_pin,
                                                                             motor1_counterclockwise_pin=motor1_counterclockwise_pin,
                                                                             motor2_clockwise_pin=motor2_clockwise_pin,
                                                                             motor2_counterclockwise_pin=motor2_counterclockwise_pin,
                                                                             default_movement_time=default_movement_time,
                                                                             movement_mode=movement_mode)
            else:
                warn("Trying to re-instantiate a the Singleton class controller. Instantiation skipped")

        def __getattr__(self, item):
            return getattr(self.instance, item)

        def __setattr__(self, key, value):
            return setattr(self.instance, key, value)

        def __exit__(self, exc_type, exc_value, traceback):
            self.stop()
            self.cleanup()
            MotorController.instance = None

        def cleanup(self):
            """
            Clean the GPIO. Automatically called on __exit__.
            """
            GPIO.cleanup()

        def keep_movement(self, time):
            """
            Depending on the movement_mode atrribute moves the robot for a given amount of time and then stops or
            schedules a trigger for stopping the robot after time seconds and returns the control.
            :param time: Float. Amount of time (in seconds) to still moving.
            """
            if self.movement_mode == ASYNC_MODE:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time))
            elif self.movement_mode == HALF_MODE:
                sleep(time / 2)
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time / 2))
            elif self.movement_mode == ONE_THIRD_MODE:
                sleep(time * 0.33)
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time *0.66))
            elif self.movement_mode == SYNC_MODE:
                sleep(time)
                self.stop()
            else:
                raise ValueError("Mode {mode} does not exist".format(mode=self.movement_mode))

        def stop(self):
            """
            Send a signal to all motors for stopping the movement
            """
            GPIO.output(self.motor_right_clockwise, False)
            GPIO.output(self.motor_left_clockwise, False)
            GPIO.output(self.motor_right_counterclockwise, False)
            GPIO.output(self.motor_left_counterclockwise, False)

        def idle(self, time=None):
            """
            Stay idle for a given amount of time.
            :param time: Float. Amount of time (in seconds) to still moving.
            """
            while self.movement_mode in [ASYNC_MODE, HALF_MODE, ONE_THIRD_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            self.stop()
            self.keep_movement(time=time)



        def move_straight(self, front=True, time=None):
            """
            Moves in a straight line for a given amount of time.
            :param front: Boolean. If True moves forward, if False moves backwards.
            :param time: Float. Amount of time (in seconds) to still moving.
            """
            while self.movement_mode in [ASYNC_MODE, HALF_MODE, ONE_THIRD_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            GPIO.output(self.motor_right_clockwise, front)
            GPIO.output(self.motor_left_clockwise, front)
            GPIO.output(self.motor_right_counterclockwise, not front)
            GPIO.output(self.motor_left_counterclockwise, not front)
            self.keep_movement(time=time)

        def rotate(self, clockwise=True, time=None):
            """
            Rotate (over its central axis) for a given amount of time
            :param clockwise: Boolean. If True rotates clockwise, if False rotates counter clockwise.
            :param time: Float. Amount of time (in seconds) to still moving.
            """
            while self.movement_mode in [ASYNC_MODE, HALF_MODE, ONE_THIRD_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            GPIO.output(self.motor_right_clockwise, clockwise)
            GPIO.output(self.motor_left_clockwise, not clockwise)
            GPIO.output(self.motor_right_counterclockwise, not clockwise)
            GPIO.output(self.motor_left_counterclockwise, clockwise)
            self.keep_movement(time=time)

        def turn(self, right=True, front=True, time=None):
            """
            Turns in one direction (moves only one wheel, so implicitly includes a slight vertical movement).
            :param right: Boolean. If True turns right, if False turns left.
            :param front: Boolean. If True moves forward, if False moves backwards.
            :param time: Float. Amount of time (in seconds) to still moving.
            """
            while self.movement_mode in [ASYNC_MODE, HALF_MODE, ONE_THIRD_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            GPIO.output(self.motor_right_clockwise, front and right) #11
            GPIO.output(self.motor_left_clockwise, front and not right) # 10
            GPIO.output(self.motor_right_counterclockwise, not front and not right) #00
            GPIO.output(self.motor_left_counterclockwise, not front and right) #01
            self.keep_movement(time=time)
except:
    warn("GPIO module not found. MotorController will be a Mock Object")
    """
    Do a Mock Object for testing without the Raspberry
    """
    class MotorController:
        def __init__(self, motor1_clockwise_pin=MOTOR1_CLOCKWISE, motor1_counterclockwise_pin=MOTOR1_COUNTERCLOCKWISE,
                     motor2_clockwise_pin=MOTOR2_CLOCKWISE, motor2_counterclockwise_pin=MOTOR2_COUNTERCLOCKWISE,
                     default_movement_time=MOVEMENT_TIME, movement_mode = DEFAULT_MOVEMENT_MODE):
            self.scheduler = BackgroundScheduler(daemon=True)
            self.scheduler.start()
            self.executed_jobs = 0
            self.movement_mode = movement_mode

        def keep_movement(self, time):
            if self.movement_mode == ASYNC_MODE:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time))
            elif self.movement_mode == HALF_MODE:
                sleep(time / 2)
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time / 2))
            elif self.movement_mode == ONE_THIRD_MODE:
                sleep(time * 0.33)
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time * 0.66))
            elif self.movement_mode == SYNC_MODE:
                sleep(time)
                self.stop()
            else:
                raise ValueError("Mode {mode} does not exist".format(mode=self.movement_mode))

        def idle(self, time=None):
            while self.movement_mode in [ASYNC_MODE, HALF_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            self.keep_movement(time=0.1)

        def move_straight(self, front=True, time=None):
            while self.movement_mode in [ASYNC_MODE, HALF_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            self.keep_movement(time=0.1)

        def rotate(self, clockwise=True, time=None):
            while self.movement_mode in [ASYNC_MODE, HALF_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            self.keep_movement(time=0.1)

        def turn(self, right=True, front = True, time=None):
            while self.movement_mode in [ASYNC_MODE, HALF_MODE] and len(self.scheduler.get_jobs()) != 0: pass
            self.keep_movement(time=0.1)

        def stop(self):
            self.executed_jobs+=1