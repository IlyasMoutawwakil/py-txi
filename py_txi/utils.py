import socket
import subprocess
from json import loads


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


LEVEL_TO_COLOR = {
    "DEBUG": "0;34m",
    "INFO": "0;32m",
    "WARNING": "0;33m",
    "WARN": "0;33m",
    "ERROR": "0;31m",
    "CRITICAL": "0;31m",
}


def color_text(text: str, color: str) -> str:
    return f"\033[{color}{text}\033[0m"


def colored_json_logs(log: str) -> str:
    dict_log = loads(log)

    fields = dict_log.get("fields", {})
    level = dict_log.get("level", "could not parse level")
    target = dict_log.get("target", "could not parse target")
    timestamp = dict_log.get("timestamp", "could not parse timestamp")
    message = fields.get("message", dict_log.get("message", "could not parse message"))

    color = LEVEL_TO_COLOR.get(level, "0;37m")

    level = color_text(level, color)
    message = color_text(message, color)

    return f"[{timestamp}][{level}][{target}] - {message}"
