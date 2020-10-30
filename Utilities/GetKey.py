import sys
import signal
import os

TIMEOUT = 3

def interrupt_enter():
    pass

try:
    # For Linux
    signal.signal(signal.SIGALRM, interrupt_enter)
    import tty
    import termios
    def get_key(timeout=TIMEOUT):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            signal.alarm(timeout)
            tty.setraw(sys.stdin.fileno())
            key = int(sys.stdin.read(1))
        except:
            key = -1
        finally:
            signal.alarm(0)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

except:
    # For testing in Windows (Mock object
    def get_key(timeout=TIMEOUT):
        return -1