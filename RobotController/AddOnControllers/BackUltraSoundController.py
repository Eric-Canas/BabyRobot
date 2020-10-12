from time import sleep, time
from warnings import warn

ECHO = 2
TRIG = 3
SOUND_M_BY_SEC = 343
SIGNAL_TIME = 1e-05
try:
    import RPi.GPIO as GPIO
    class BackUltraSoundController:
        class __BackUltraSoundController:
            def __init__(self, echo_pin = ECHO, trigger_pin = TRIG):
                self.echo = echo_pin
                self.trigger = trigger_pin

                GPIO.setmode(GPIO.BCM)

                GPIO.setup(self.trigger, GPIO.OUT)
                GPIO.setup(self.echo, GPIO.IN)
                GPIO.output(self.trigger, True)
        instance = None

        def __init__(self, echo_pin = ECHO, trigger_pin = TRIG):
            if BackUltraSoundController.instance is None:
                BackUltraSoundController.instance = BackUltraSoundController.__BackUltraSoundController(echo_pin=echo_pin,
                                                                                                        trigger_pin=trigger_pin)
            else:
                warn("Trying to re-instantiate a the Singleton class controller. Instantiation skipped")

        def __exit__(self, exc_type, exc_value, traceback):
            self.cleanup()
            BackUltraSoundController.instance = None

        def cleanup(self):
            GPIO.cleanup()


        def get_distance(self):
            # Send the signal
            GPIO.output(self.trigger, True)
            sleep(1e-05)
            GPIO.output(self.trigger, False)

            # Wait to the first pulse
            first_pulse = time()
            while not GPIO.input(self.echo):
                first_pulse = time()

            end_pulse = time()
            while GPIO.input(self.echo):
                end_pulse = time()

            duration_in_seconds = end_pulse - first_pulse
            # measure de distance as: Distance traversed by sound in duration seconds between 2 (it went and returned)
            distance_in_m = (SOUND_M_BY_SEC * duration_in_seconds) / 2

            return distance_in_m
except:
    warn("GPIO not found. BackUltraSoundController will be a Mock Object")
    """
    If raspberry is not reachable make a Mock Object for testing.
    """
    from random import uniform
    class BackUltraSoundController:
        def __init__(self, echo_pin=ECHO, trigger_pin=TRIG):
            pass

        def get_distance(self):
            return uniform(.1, 5.)