from time import sleep, time
from warnings import warn

ECHO = 5
TRIG = 6
SOUND_M_BY_SEC = 343
SIGNAL_TIME = 1e-05
TIMEOUT = 0.05
TIMEOUT_CONSTANT = 5.
try:
    import RPi.GPIO as GPIO
    class BackUltraSoundController:
        class __BackUltraSoundController:
            def __init__(self, echo_pin = ECHO, trigger_pin = TRIG, timeout = TIMEOUT):
                self.echo = echo_pin
                self.trigger = trigger_pin
                self.timeout = timeout

                GPIO.setmode(GPIO.BCM)

                GPIO.setup(self.trigger, GPIO.OUT)
                GPIO.setup(self.echo, GPIO.IN)
                GPIO.output(self.trigger, True)
                sleep(SIGNAL_TIME*100)
                GPIO.output(self.trigger, False)
                sleep(1.)

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

        def __getattr__(self, item):
            return getattr(self.instance, item)

        def __setattr__(self, key, value):
            return setattr(self.instance, key, value)

        def cleanup(self):
            GPIO.cleanup()


        def get_distance(self):
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
    warn("GPIO not found. BackUltraSoundController will be a Mock Object")
    """
    If raspberry is not reachable make a Mock Object for testing.
    """
    from random import uniform
    class BackUltraSoundController:
        def __init__(self, echo_pin=ECHO, trigger_pin=TRIG, timeout=TIMEOUT):
            pass

        def get_distance(self):
            return uniform(.1, 5.)