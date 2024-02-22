import os
import re
from abc import ABC
from logging import INFO, basicConfig, getLogger
from typing import Any, Dict, List, Optional, Union

import docker
import docker.errors
import docker.types

from .utils import HF_CACHE_DIR

basicConfig(level=INFO)


DOCKER = docker.from_env()
LOGGER = getLogger("inference-server")


class InferenceServer(ABC):
    NAME: str = "Inference-Server"

    def __init__(
        self,
        # model options
        model: str,
        revision: str,
        # image options
        image: str,
        # docker options
        port: int = 1111,
        shm_size: str = "1g",
        address: str = "127.0.0.1",
        volumes: Dict[str, Any] = {HF_CACHE_DIR: "/data"},  # connects local hf cache to /data folder
        gpus: Optional[Union[str, int]] = None,  # e.g. "all" or "0,1,2,3" or 4 for NVIDIA
        devices: Optional[List[str]] = None,  # e.g. ["/dev/kfd", "/dev/dri"] for ROCm
        # launcher options
        **kwargs,
    ) -> None:
        # model options
        self.model = model
        self.revision = revision
        # docker options
        self.port = port
        self.image = image
        self.volumes = volumes
        self.address = address
        self.shm_size = shm_size
        # device options
        self.gpus = gpus
        self.devices = devices

        try:
            LOGGER.info(f"\t+ Checking if {self.NAME} image is available locally")
            DOCKER.images.get(self.image)
            LOGGER.info(f"\t+ {self.NAME} image found locally")
        except docker.errors.ImageNotFound:
            LOGGER.info(f"\t+ {self.NAME} image not found locally, pulling from Docker Hub")
            DOCKER.images.pull(self.image)

        LOGGER.info(f"\t+ Building {self.NAME} URL")
        self.build_url()

        LOGGER.info(f"\t+ Building {self.NAME} environment")
        self.build_env()

        LOGGER.info(f"\t+ Building {self.NAME} devices")
        self.build_devices()

        LOGGER.info(f"\t+ Building {self.NAME} command")
        self.build_command()

        LOGGER.info(f"\t+ Running {self.NAME} server")
        self.run_container()

        LOGGER.info(f"\t+ Waiting for {self.NAME} server to be ready")
        self.wait()

        LOGGER.info(f"\t+ Connecting to {self.NAME} server")
        self.connect_client()

    def run_container(self):
        self.container = DOCKER.containers.run(
            image=self.image,
            command=self.command,
            shm_size=self.shm_size,
            ports={"80/tcp": (self.address, self.port)},
            volumes={source: {"bind": target, "mode": "rw"} for source, target in self.volumes.items()},
            device_requests=self.device_requests,
            devices=self.devices,
            environment=self.env,
            auto_remove=True,
            detach=True,
        )

    def wait(self):
        raise NotImplementedError

    def connect_client(self):
        raise NotImplementedError

    def build_devices(self):
        if self.gpus is not None and isinstance(self.gpus, str) and self.gpus == "all":
            LOGGER.info("\t+ Using all GPU(s)")
            self.device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        elif self.gpus is not None and isinstance(self.gpus, int):
            LOGGER.info(f"\t+ Using {self.gpus} GPU(s)")
            self.device_requests = [docker.types.DeviceRequest(count=self.gpus, capabilities=[["gpu"]])]
        elif self.gpus is not None and isinstance(self.gpus, str) and re.match(r"^\d+(,\d+)*$", self.gpus):
            LOGGER.info(f"\t+ Using GPU(s) {self.gpus}")
            self.device_requests = [docker.types.DeviceRequest(device_ids=[self.gpus], capabilities=[["gpu"]])]
        else:
            LOGGER.info("\t+ Not using any GPU(s)")
            self.device_requests = None

        if self.devices is not None and isinstance(self.devices, list) and all(os.path.exists(d) for d in self.devices):
            LOGGER.info(f"\t+ Using custom device(s) {self.devices}")
            self.devices = self.devices
        else:
            LOGGER.info("\t+ Not using any custom device(s)")
            self.devices = None

    def build_url(self):
        self.url = f"http://{self.address}:{self.port}"

    def build_env(self):
        self.env = {}

    def build_command(self):
        self.command = []

    def close(self) -> None:
        if hasattr(self, "container"):
            LOGGER.info("\t+ Stoping Docker container")
            self.container.stop()
            self.container.wait()
            LOGGER.info("\t+ Docker container stopped")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
