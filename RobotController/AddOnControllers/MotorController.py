from time import sleep
from warnings import warn
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from RobotController.RLConstants import MOVEMENT_TIME

MOTOR1_CLOCKWISE = 27
MOTOR1_COUNTERCLOCKWISE = 17

MOTOR2_CLOCKWISE = 24
MOTOR2_COUNTERCLOCKWISE = 23

try:
    import RPi.GPIO as GPIO
    class MotorController:
        class __MotorController:
            def __init__(self, motor1_clockwise_pin = MOTOR1_CLOCKWISE, motor1_counterclockwise_pin = MOTOR1_COUNTERCLOCKWISE,
                         motor2_clockwise_pin = MOTOR2_CLOCKWISE, motor2_counterclockwise_pin = MOTOR2_COUNTERCLOCKWISE,
                         default_movement_time =MOVEMENT_TIME, asynchronous = False):

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
                self.asynchronous = asynchronous
                if self.asynchronous:
                    self.scheduler = BackgroundScheduler(daemon=True)
                    self.scheduler.start()


        instance = None

        def __init__(self, motor1_clockwise_pin = MOTOR1_CLOCKWISE, motor1_counterclockwise_pin = MOTOR1_COUNTERCLOCKWISE,
                     motor2_clockwise_pin = MOTOR2_CLOCKWISE, motor2_counterclockwise_pin = MOTOR2_COUNTERCLOCKWISE,
                     default_movement_time =MOVEMENT_TIME, asynchronous = False):
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
            self.stop()
            self.cleanup()
            MotorController.instance = None

        def cleanup(self):
            GPIO.cleanup()

        def stop(self):
            GPIO.output(self.motor_left_clockwise, False)
            GPIO.output(self.motor_left_counterclockwise, False)
            GPIO.output(self.motor_right_clockwise, False)
            GPIO.output(self.motor_right_counterclockwise, False)

        def idle(self, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time))
            else:
                sleep(time)


        def move_straight(self, front=True, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            GPIO.output(self.motor_left_clockwise, front)
            GPIO.output(self.motor_left_counterclockwise, not front)
            GPIO.output(self.motor_right_clockwise, front)
            GPIO.output(self.motor_right_counterclockwise, not front)
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time))
            else:
                sleep(time)

        def rotate(self, clockwise=True, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            GPIO.output(self.motor_left_clockwise, clockwise)
            GPIO.output(self.motor_left_counterclockwise, not clockwise)
            GPIO.output(self.motor_right_clockwise, not clockwise)
            GPIO.output(self.motor_right_counterclockwise, clockwise)
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time))
            else:
                sleep(time)

        def turn(self, right=True, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            time = time if time is not None else self.default_movement_time
            if right:
                GPIO.output(self.motor_left_clockwise, True)
                GPIO.output(self.motor_left_counterclockwise, False)
                GPIO.output(self.motor_right_clockwise, False)
                GPIO.output(self.motor_right_counterclockwise, False)
            else:
                GPIO.output(self.motor_left_clockwise, False)
                GPIO.output(self.motor_left_counterclockwise, False)
                GPIO.output(self.motor_right_clockwise, True)
                GPIO.output(self.motor_right_counterclockwise, False)
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=time))
            else:
                sleep(time)
except:
    warn("GPIO module not found. MotorController will be a Mock Object")
    """
    Do a Mock Object for testing without the Raspberry
    """
    class MotorController:
        def __init__(self, motor1_clockwise_pin=MOTOR1_CLOCKWISE, motor1_counterclockwise_pin=MOTOR1_COUNTERCLOCKWISE,
                     motor2_clockwise_pin=MOTOR2_CLOCKWISE, motor2_counterclockwise_pin=MOTOR2_COUNTERCLOCKWISE,
                     default_movement_time=MOVEMENT_TIME, asynchronous = False):
            self.scheduler = BackgroundScheduler(daemon=True)
            self.scheduler.start()
            self.executed_jobs = 0
            self.asynchronous = asynchronous

        def idle(self, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=0.01))
            else:
                sleep(0.01)
                self.executed_jobs += 1

        def move_straight(self, front=True, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=0.01))
            else:
                sleep(0.01)
                self.executed_jobs += 1

        def rotate(self, clockwise=True, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=0.01))
            else:
                sleep(0.01)
                self.executed_jobs += 1

        def turn(self, right=True, time=None):
            while self.asynchronous and len(self.scheduler.get_jobs()) != 0: pass
            if self.asynchronous:
                self.scheduler.add_job(self.stop, trigger='date', run_date=datetime.now() + timedelta(seconds=0.01))
            else:
                sleep(0.01)
                self.executed_jobs += 1

        def stop(self):
            self.executed_jobs+=1