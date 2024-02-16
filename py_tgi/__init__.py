import os
import time
import subprocess
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Literal, List, Union
from contextlib import contextmanager
import signal


import docker
import docker.types
import docker.errors
from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationResponse

LOGGER = getLogger("tgi")
HF_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/huggingface/hub"

Quantization_Literal = Literal["bitsandbytes-nf4", "bitsandbytes-fp4", "gptq"]
Torch_Dtype_Literal = Literal["float32", "float16", "bfloat16"]


class TGI:
    def __init__(
        self,
        # model options
        model: str,
        revision: str = "main",
        # image options
        image: str = "ghcr.io/huggingface/text-generation-inference",
        version: str = "latest",
        # docker options
        volume: str = HF_CACHE_DIR,
        shm_size: str = "1g",
        address: str = "127.0.0.1",
        port: int = 1111,
        # tgi launcher options
        sharded: Optional[bool] = None,
        num_shard: Optional[int] = None,
        torch_dtype: Optional[Torch_Dtype_Literal] = None,
        quantization: Optional[Quantization_Literal] = None,
        trust_remote_code: Optional[bool] = False,
        disable_custom_kernels: Optional[bool] = False,
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

        LOGGER.info("\t+ Starting Docker client")
        self.docker_client = docker.from_env()

        try:
            LOGGER.info("\t+ Checking if TGI image exists")
            self.docker_client.images.get(f"{self.image}:{self.version}")
        except docker.errors.ImageNotFound:
            LOGGER.info(
                "\t+ TGI image not found, downloading it (this may take a while)"
            )
            self.docker_client.images.pull(f"{self.image}:{self.version}")

        LOGGER.info("\t+ Building TGI command")
        self.command = ["--model-id", self.model, "--revision", self.revision]

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
                LOGGER.info(
                    "\t+ `CUDA_VISIBLE_DEVICES` is set, using the specified GPU(s)"
                )
                device_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
            else:
                LOGGER.info(
                    "\t+ `CUDA_VISIBLE_DEVICES` is not set, using nvidia-smi to detect GPU(s)"
                )
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
        with timeout(60):
            for line in self.tgi_container.logs(stream=True):
                tgi_log = line.decode("utf-8").strip()
                if "Connected" in tgi_log:
                    break
                elif "Error" in tgi_log:
                    raise Exception(f"\t {tgi_log}")

                LOGGER.info(f"\t {tgi_log}")

        LOGGER.info("\t+ Conecting to TGI server")
        self.url = f"http://{self.address}:{self.port}"
        with timeout(60):
            while True:
                try:
                    self.tgi_client = InferenceClient(model=self.url)
                    self.tgi_client.text_generation("Hello world!")
                    LOGGER.info(f"\t+ Connected to TGI server at {self.url}")
                    break
                except Exception:
                    LOGGER.info("\t+ TGI server not ready, retrying in 1 second")
                    time.sleep(1)

    def close(self) -> None:
        if hasattr(self, "tgi_container"):
            LOGGER.info("\t+ Stoping TGI container")
            self.tgi_container.stop()
            LOGGER.info("\t+ Waiting for TGI container to stop")
            self.tgi_container.wait()

        if hasattr(self, "docker_client"):
            LOGGER.info("\t+ Closing docker client")
            self.docker_client.close()

    def __call__(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Union[TextGenerationResponse, List[TextGenerationResponse]]:
        return self.generate(prompt, **kwargs)

    def generate(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Union[TextGenerationResponse, List[TextGenerationResponse]]:
        if isinstance(prompt, str):
            return self.tgi_client.text_generation(prompt=prompt, **kwargs)

        elif isinstance(prompt, list):
            with ThreadPoolExecutor(max_workers=len(prompt)) as executor:
                futures = [
                    executor.submit(
                        self.tgi_client.text_generation, prompt=prompt[i], **kwargs
                    )
                    for i in range(len(prompt))
                ]

            output = []
            for i in range(len(prompt)):
                output.append(futures[i].result())
            return output


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
