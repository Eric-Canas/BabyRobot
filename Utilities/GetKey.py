import sys
import signal

from RobotController.RLConstants import REQUEST_FOR_ACTION_TIMEOUT


def interrupt_enter():
    pass

try:
    # For Linux
    signal.signal(signal.SIGALRM, interrupt_enter)
    import tty
    import termios
    def get_key(timeout=REQUEST_FOR_ACTION_TIMEOUT):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            signal.alarm(timeout)
            tty.setraw(sys.stdin.fileno())
            # Read only one number without expecting enter
            key = int(sys.stdin.read(1))
        except:
            key = -1
        finally:
            signal.alarm(0)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

except:
    # For testing in Windows (Mock object)
    def get_key(timeout=REQUEST_FOR_ACTION_TIMEOUT):
        return -1