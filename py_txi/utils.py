import socket
import subprocess
from datetime import datetime
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


LEVEL_TO_MESSAGE_STYLE = {
    "DEBUG": "\033[37m",
    "INFO": "\033[37m",
    "WARN": "\033[33m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[31m",
}
TIMESTAMP_STYLE = "\033[32m"
TARGET_STYLE = "\033[0;38m"
LEVEL_STYLE = "\033[1;30m"


def color_text(text: str, color: str) -> str:
    return f"{color}{text}\033[0m"


def styled_logs(log: str) -> str:
    try:
        dict_log = loads(log)
    except Exception:
        return log

    fields = dict_log.get("fields", {})
    level = dict_log.get("level", "could not parse level")
    target = dict_log.get("target", "could not parse target")
    timestamp = dict_log.get("timestamp", "could not parse timestamp")
    message = fields.get("message", dict_log.get("message", "could not parse message"))
    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")

    message = color_text(message, LEVEL_TO_MESSAGE_STYLE.get(level, "\033[37m"))
    timestamp = color_text(timestamp, TIMESTAMP_STYLE)
    target = color_text(target, TARGET_STYLE)
    level = color_text(level, LEVEL_STYLE)

    return f"[{timestamp}][{target}][{level}] - {message}"
