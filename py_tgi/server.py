import os
import time
import subprocess
from logging import getLogger, basicConfig, INFO
from typing import Optional

import docker
import docker.errors
import docker.types

from huggingface_hub import InferenceClient

basicConfig(level=INFO)
LOGGER = getLogger("tgi-server")
HF_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/huggingface/hub"


class TGIServer:
    def __init__(
        self,
        model: str,
        revision: str = "main",
        version: str = "latest",
        image: str = "ghcr.io/huggingface/text-generation-inference",
        volume: str = HF_CACHE_DIR,
        shm_size: str = "1g",
        address: str = "127.0.0.1",
        port: int = 1111,
        trust_remote_code: bool = False,
        disable_custom_kernels: bool = False,
        sharded: Optional[bool] = None,
        num_shard: Optional[int] = None,
        torch_dtype: Optional[str] = None,  # float32, float16, bfloat16
        quantization: Optional[str] = None,  # bitsandbytes-nf4, bitsandbytes-fp4, gptq
    ) -> None:
        # model options
        self.model = model
        self.revision = revision
        # image options
        self.image = image
        self.version = version
        # docker options
        self.port = port
        self.volume = volume
        self.address = address
        self.shm_size = shm_size
        # tgi launcher options
        self.sharded = sharded
        self.num_shard = num_shard
        self.torch_dtype = torch_dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.disable_custom_kernels = disable_custom_kernels
        self.url = f"http://{self.address}:{self.port}"

        LOGGER.info("\t+ Starting Docker client")
        self.docker_client = docker.from_env()

        try:
            LOGGER.info("\t+ Checking if TGI image exists")
            self.docker_client.images.get(f"{self.image}:{self.version}")
        except docker.errors.ImageNotFound:
            LOGGER.info("\t+ TGI image not found, pulling it")
            self.docker_client.images.pull(f"{self.image}:{self.version}")

        LOGGER.info("\t+ Building TGI command")
        self.command = [
            "--model-id",
            self.model,
            "--revision",
            self.revision,
        ]

        if self.torch_dtype is not None:
            self.command.extend(["--torch-dtype", self.torch_dtype])
        if self.quantization is not None:
            self.command.extend(["--quantize", self.quantization])
        if self.sharded is not None:
            self.command.extend(["--sharded", str(self.sharded).lower()])
        if self.num_shard is not None:
            self.command.extend(["--num-shard", str(self.num_shard)])
        if self.trust_remote_code:
            self.command.append("--trust-remote-code")
        if self.disable_custom_kernels:
            self.command.append("--disable-custom-kernels")

        try:
            LOGGER.info("\t+ Checking if GPU is available")
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                LOGGER.info("\t+ Using GPU(s) from CUDA_VISIBLE_DEVICES")
                device_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
            else:
                device_ids = ",".join([str(device) for device in get_gpu_devices()])
                LOGGER.info("\t+ Using GPU(s) from nvidia-smi")

            LOGGER.info(f"\t+ Using GPU(s): {device_ids}")
            self.device_requests = [
                docker.types.DeviceRequest(
                    driver="nvidia",
                    device_ids=[str(device_ids)],
                    capabilities=[["gpu"]],
                )
            ]
        except Exception:
            LOGGER.info("\t+ No GPU detected")
            self.device_requests = None

        self.tgi_container = self.docker_client.containers.run(
            image=f"{self.image}:{self.version}",
            command=self.command,
            shm_size=self.shm_size,
            volumes={self.volume: {"bind": "/data", "mode": "rw"}},
            ports={"80/tcp": (self.address, self.port)},
            device_requests=self.device_requests,
            detach=True,
        )

        LOGGER.info("\t+ Waiting for TGI server to be ready")
        for line in self.tgi_container.logs(stream=True):
            tgi_log = line.decode("utf-8").strip()
            if not tgi_log:
                continue
            elif "Connected" in tgi_log:
                break
            else:
                LOGGER.info(f"\t {tgi_log}")

        while True:
            try:
                dummy_client = InferenceClient(model=self.url)
                dummy_client.text_generation("Hello world!")
                del dummy_client
                break
            except Exception:
                LOGGER.info("\t+ Couldn't connect to TGI server")
                LOGGER.info("\t+ Retrying in 0.1s")
                time.sleep(0.1)

        LOGGER.info(f"\t+ TGI server ready at {self.url}")

    def close(self) -> None:
        if hasattr(self, "tgi_container"):
            LOGGER.info("\t+ Stoping TGI container")
            self.tgi_container.stop()
            LOGGER.info("\t+ Waiting for TGI container to stop")
            self.tgi_container.wait()

        if hasattr(self, "docker_client"):
            LOGGER.info("\t+ Closing docker client")
            self.docker_client.close()


def get_gpu_devices():
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

    return device_ids
