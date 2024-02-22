import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import List, Literal, Optional, Union

import docker
import docker.errors
import docker.types
from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationResponse

from .utils import is_rocm_system

basicConfig(level=INFO)

CONNECTION_TIMEOUT = 60
LOGGER = getLogger("py-tgi")
HF_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/huggingface/hub"


DType_Literal = Literal["float32", "float16", "bfloat16"]
Quantize_Literal = Literal["bitsandbytes-nf4", "bitsandbytes-fp4", "gptq"]


class TGI:
    def __init__(
        self,
        # model options
        model: str,
        revision: str = "main",
        # image options
        image: str = "ghcr.io/huggingface/text-generation-inference:latest",
        # docker options
        port: int = 1111,
        shm_size: str = "1g",
        address: str = "127.0.0.1",
        volume: str = HF_CACHE_DIR,  # automatically connects local hf cache to the container's /data
        devices: Optional[List[str]] = None,  # ["/dev/kfd", "/dev/dri"] or None for custom devices (ROCm)
        gpus: Optional[Union[str, int]] = None,  # "all" or "0,1,2,3" or 3 or None for NVIDIA
        # tgi launcher options
        sharded: Optional[bool] = None,
        num_shard: Optional[int] = None,
        dtype: Optional[DType_Literal] = None,
        quantize: Optional[Quantize_Literal] = None,
        trust_remote_code: Optional[bool] = False,
        disable_custom_kernels: Optional[bool] = False,
    ) -> None:
        # model options
        self.model = model
        self.revision = revision
        # image options
        self.image = image
        # docker options
        self.port = port
        self.volume = volume
        self.address = address
        self.shm_size = shm_size
        # tgi launcher options
        self.dtype = dtype
        self.sharded = sharded
        self.num_shard = num_shard
        self.quantize = quantize
        self.trust_remote_code = trust_remote_code
        self.disable_custom_kernels = disable_custom_kernels

        if is_rocm_system() and "-rocm" not in self.image:
            LOGGER.warning("ROCm system detected, but the image does not contain '-rocm'. Adding it.")
            self.image = self.image + "-rocm"

        LOGGER.info("\t+ Starting Docker client")
        self.docker_client = docker.from_env()

        try:
            LOGGER.info("\t+ Checking if TGI image is available locally")
            self.docker_client.images.get(self.image)
            LOGGER.info("\t+ TGI image found locally")
        except docker.errors.ImageNotFound:
            LOGGER.info("\t+ TGI image not found locally, pulling from Docker Hub")
            self.docker_client.images.pull(self.image)

        env = {}
        if os.environ.get("HUGGING_FACE_HUB_TOKEN", None) is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

        LOGGER.info("\t+ Building TGI command")
        self.command = ["--model-id", self.model, "--revision", self.revision]

        if self.sharded is not None:
            self.command.extend(["--sharded", str(self.sharded).lower()])
        if self.num_shard is not None:
            self.command.extend(["--num-shard", str(self.num_shard)])
        if self.quantize is not None:
            self.command.extend(["--quantize", self.quantize])
        if self.dtype is not None:
            self.command.extend(["--dtype", self.dtype])

        if self.trust_remote_code:
            self.command.append("--trust-remote-code")
        if self.disable_custom_kernels:
            self.command.append("--disable-custom-kernels")

        if gpus is not None and isinstance(gpus, str) and gpus == "all":
            LOGGER.info("\t+ Using all GPU(s)")
            self.device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        elif gpus is not None and isinstance(gpus, int):
            LOGGER.info(f"\t+ Using {gpus} GPU(s)")
            self.device_requests = [docker.types.DeviceRequest(count=gpus, capabilities=[["gpu"]])]
        elif gpus is not None and isinstance(gpus, str) and re.match(r"^\d+(,\d+)*$", gpus):
            LOGGER.info(f"\t+ Using GPU(s) {gpus}")
            self.device_requests = [docker.types.DeviceRequest(device_ids=[gpus], capabilities=[["gpu"]])]
        else:
            LOGGER.info("\t+ Not using any GPU(s)")
            self.device_requests = None

        if devices is not None and isinstance(devices, list) and all(Path(d).exists() for d in devices):
            LOGGER.info(f"\t+ Using custom device(s) {devices}")
            self.devices = devices
        else:
            LOGGER.info("\t+ Not using any custom device(s)")
            self.devices = None

        self.closed = False
        self.docker_container = self.docker_client.containers.run(
            image=self.image,
            command=self.command,
            volumes={self.volume: {"bind": "/data", "mode": "rw"}},
            ports={"80/tcp": (self.address, self.port)},
            device_requests=self.device_requests,
            shm_size=self.shm_size,
            devices=self.devices,
            environment=env,
            auto_remove=True,  # this is so cool
            detach=True,
        )

        LOGGER.info("\t+ Waiting for TGI server to be ready")
        for line in self.docker_container.logs(stream=True):
            tgi_log = line.decode("utf-8").strip()
            if "Connected" in tgi_log:
                LOGGER.info(f"\t {tgi_log}")
                break
            elif "Error" in tgi_log:
                LOGGER.info(f"\t {tgi_log}")
                raise Exception("TGI server failed to start")
            else:
                LOGGER.info(f"\t {tgi_log}")

        LOGGER.info("\t+ Conecting to TGI server")
        self.url = f"http://{self.address}:{self.port}"

        start_time = time.time()
        while time.time() - start_time < CONNECTION_TIMEOUT:
            try:
                self.tgi_client = InferenceClient(model=self.url)
                self.tgi_client.text_generation("Hello world!")
                LOGGER.info("\t+ Connected to TGI server successfully")
                return
            except Exception:
                LOGGER.info("\t+ TGI server is not ready yet, waiting 1 second")
                time.sleep(1)

        raise Exception("TGI server took too long to start (60 seconds)")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def generate(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Union[TextGenerationResponse, List[TextGenerationResponse]]:
        if isinstance(prompt, str):
            return self.tgi_client.text_generation(prompt=prompt, **kwargs)

        elif isinstance(prompt, list):
            with ThreadPoolExecutor(max_workers=len(prompt)) as executor:
                futures = [
                    executor.submit(self.tgi_client.text_generation, prompt=prompt[i], **kwargs)
                    for i in range(len(prompt))
                ]

            output = []
            for i in range(len(prompt)):
                output.append(futures[i].result())
            return output

    def close(self) -> None:
        if not self.closed:
            if hasattr(self, "docker_container"):
                LOGGER.info("\t+ Stoping docker container")
                self.docker_container.stop()
                self.docker_container.wait()

            if hasattr(self, "docker_client"):
                LOGGER.info("\t+ Closing docker client")
                self.docker_client.close()

        self.closed = True

    def __call__(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Union[TextGenerationResponse, List[TextGenerationResponse]]:
        return self.generate(prompt, **kwargs)
