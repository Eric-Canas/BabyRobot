import sys
import signal

from RobotController.RLConstants import REQUEST_FOR_ACTION_TIMEOUT


def interrupt_enter():
    pass

try:
    """
    When executing in the raspberry (or a Linux system)
    """
    # For Linux
    signal.signal(signal.SIGALRM, interrupt_enter)
    import tty
    import termios
    def get_key(timeout=REQUEST_FOR_ACTION_TIMEOUT):
        """
        Return an input introduced by the user in the keyboard, or -1 if he/she did not introduce any input in timeout
        seconds
        :param timeout: Float. Seconds for which the keyboard is listening for a user input
        :return: Int. The Input introduced by the user in the keyboard, or -1 if he/she did not introduce any input.
        seconds
        """
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
    """
    When executing in windows it is a Mock Object
    """
    # For testing in Windows (Mock object)
    def get_key(timeout=REQUEST_FOR_ACTION_TIMEOUT):
        return -1