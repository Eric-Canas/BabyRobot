"""
If the code is executed in a Raspberry, define UltrasoundController as the communicator with the SR04.
Elsewhere instantiate a Mock Object for debugging in the PC.
"""
from time import sleep, time
from warnings import warn

BACK_ECHO = 4
BACK_TRIGGER = 18
FRONT_ECHO = 20
FRONT_TRIGGER = 21
SOUND_M_BY_SEC = 343
SIGNAL_TIME = 1e-05
TIMEOUT = 0.05
TIMEOUT_CONSTANT = 5.
try:
    import RPi.GPIO as GPIO
    class UltraSoundController:
        def __init__(self, echo_pin = BACK_ECHO, trigger_pin = BACK_TRIGGER, timeout = TIMEOUT):
            """
            Communicator with the ultrasounds that allows to read distances
            :param echo_pin: Int. Echo pin of the SR04
            :param trigger_pin: Int. Trigger pin of the SR04
            :param timeout: Float. Time out time (in seconds) for considering that the echo signal will not return.
            """
            self.echo = echo_pin
            self.trigger = trigger_pin
            self.timeout = timeout

            GPIO.setmode(GPIO.BCM)

            GPIO.setup(self.trigger, GPIO.OUT)
            GPIO.setup(self.echo, GPIO.IN)
            GPIO.output(self.trigger, True)
            sleep(SIGNAL_TIME * 100)
            GPIO.output(self.trigger, False)
            sleep(1.)

        def __exit__(self, exc_type, exc_value, traceback):
            self.cleanup()

        def cleanup(self):
            """
            Clean the GPIO. Automatically called on __exit__.
            """
            GPIO.cleanup()


        def get_distance(self):
            """
            Measures wich is the distance to the closer obstacle.
            :return: Float. Distance in cm to the closer obstacle
            """
            # Send the signal
            GPIO.output(self.trigger, True)
            sleep(SIGNAL_TIME)
            GPIO.output(self.trigger, False)

            # Wait to the first pulse
            first_pulse = time()
            timeout = time() + self.timeout
            while not GPIO.input(self.echo):
                if time() > timeout:
                    print("ULTRASOUND: Echo input never received")
                    return TIMEOUT_CONSTANT
                else:
                    first_pulse = time()

            end_pulse = time()
            timeout = time() + self.timeout
            while GPIO.input(self.echo):
                if time() > timeout:
                    print("ULTRASOUND: Echo input was received eternally")
                    return TIMEOUT_CONSTANT
                else:
                    end_pulse = time()

            duration_in_seconds = end_pulse - first_pulse
            # measure de distance as: Distance traversed by sound in duration seconds between 2 (it went and returned)
            distance_in_m = (SOUND_M_BY_SEC * duration_in_seconds) / 2

            return distance_in_m
except:
    warn("GPIO not found. UltraSoundController will be a Mock Object")
    """
    If raspberry is not reachable make a Mock Object for testing.
    """
    from random import uniform
    class UltraSoundController:
        def __init__(self, echo_pin=BACK_ECHO, trigger_pin=BACK_TRIGGER, timeout=TIMEOUT):
            pass

        def get_distance(self):
            return uniform(.1, 5.)