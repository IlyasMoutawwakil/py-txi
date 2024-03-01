import socket
import subprocess


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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
