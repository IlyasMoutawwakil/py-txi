import signal
import subprocess
from contextlib import contextmanager


def is_rocm_system() -> bool:
    try:
        subprocess.check_output(["rocm-smi"])
        return True
    except FileNotFoundError:
        return False


def is_nvidia_system() -> bool:
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except FileNotFoundError:
        return False


@contextmanager
def timeout(time: int):
    """
    Timeout context manager. Raises TimeoutError if the code inside the context manager takes longer than `time` seconds to execute.
    """

    def signal_handler(signum, frame):
        raise TimeoutError("Timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(time)

    try:
        yield
    finally:
        signal.alarm(0)
