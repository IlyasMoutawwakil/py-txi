import signal
import subprocess
from contextlib import contextmanager


def get_nvidia_gpu_devices() -> str:
    nvidia_smi = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,gpu_name,compute_cap",
                "--format=csv",
            ],
        )
        .decode("utf-8")
        .strip()
        .split("\n")[1:]
    )
    device = [
        {
            "id": int(gpu.split(", ")[0]),
            "name": gpu.split(", ")[1],
            "compute_cap": gpu.split(", ")[2],
        }
        for gpu in nvidia_smi
    ]
    device_ids = [gpu["id"] for gpu in device if "Display" not in gpu["name"]]
    device_ids = ",".join([str(device_id) for device_id in device_ids])

    return device_ids


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
