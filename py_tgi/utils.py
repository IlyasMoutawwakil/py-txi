import os
import subprocess

HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
CONNECTION_TIMEOUT = 60


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
