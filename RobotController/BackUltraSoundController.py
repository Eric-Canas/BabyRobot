import RPi.GPIO as GPIO
from time import sleep, time

ECHO = 2
TRIG = 3
SOUND_CM_BY_SEC = 343*100
SIGNAL_TIME = 1e-05

class BackUltraSoundController:
    def __init__(self, echo = ECHO, trigger = TRIG):
        self.echo = echo
        self.trigger = trigger

        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.trigger, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)
        GPIO.output(self.trigger, True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

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
        distance_in_cm = (SOUND_CM_BY_SEC * duration_in_seconds) / 2

        return distance_in_cm