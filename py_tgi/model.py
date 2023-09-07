import re
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import InferenceClient

import docker
import docker.errors
import docker.types


class TGIModel:
    def __init__(
        self,
        model: str,
        revision: str = "main",
        version: str = "latest",
        image: str = "ghcr.io/huggingface/text-generation-inference",
        volume: str = f"{os.path.expanduser('~')}/.cache/huggingface/hub",
        shm_size: str = "1g",
        address: str = "localhost",
        port: int = 1111,
        trust_remote_code: bool = False,
        disable_custom_kernels: bool = False,
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
        self.torch_dtype = torch_dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.disable_custom_kernels = disable_custom_kernels

        print("\t+ Starting Docker client")
        self.docker_client = docker.from_env()

        try:
            print("\t+ Checking if TGI image exists")
            self.docker_client.images.get(f"{self.image}:{self.version}")
        except docker.errors.ImageNotFound:
            print("\t+ TGI image not found, pulling it")
            self.docker_client.images.pull(f"{self.image}:{self.version}")

        print("\t+ Building TGI command")
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
        if self.trust_remote_code:
            self.command.append("--trust-remote-code")
        if self.disable_custom_kernels:
            self.command.append("--disable-custom-kernels")

        try:
            device_ids = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
                )
                .decode("utf-8")
                .strip()
                .replace("\n", ",")
            )
            print(f"\t+ Starting TGI container on NVIDIA device(s): {device_ids}")
            device_requests = [
                docker.types.DeviceRequest(
                    driver="nvidia",
                    device_ids=[str(device_ids)],
                    capabilities=[["gpu"]],
                )
            ]
        except Exception:
            print("\t+ Starting TGI container on CPU")
            device_requests = None

        self.tgi_container = self.docker_client.containers.run(
            image=f"{self.image}:{self.version}",
            command=self.command,
            shm_size=self.shm_size,
            volumes={self.volume: {"bind": "/data", "mode": "rw"}},
            ports={"80/tcp": (self.address, self.port)},
            device_requests=device_requests,
            detach=True,
        )

        print("\t+ Waiting for TGI server to be ready")
        for line in self.tgi_container.logs(stream=True):
            tgi_log = line.decode("utf-8").strip()
            if not tgi_log:
                continue
            elif "Connected" in tgi_log:
                print("\t+ TGI server is ready")
                break
            else:
                print(f"\t {tgi_log}")

        print("\t+ Creating InferenceClient")
        self.tgi_client = InferenceClient(model=f"http://{self.address}:{self.port}")

    def generate(
        self, prompt: Union[str, List[str]], **kwargs: Dict[str, Any]
    ) -> Union[str, List[str]]:
        if isinstance(prompt, str):
            return self.tgi_client.text_generation(
                prompt=prompt, **kwargs
            ).generated_text

        elif isinstance(prompt, list):
            with ThreadPoolExecutor(max_workers=len(input["prompt"])) as executor:
                futures = [
                    executor.submit(
                        self.tgi_client.text_generation,
                        prompt=prompt[i],
                        details=True,
                        **kwargs,
                    )
                    for i in range(len(prompt))
                ]

            output = []
            for i in range(len(input["prompt"])):
                output.append(futures[i].result().generated_text)
            return output

    def close(self) -> None:
        if hasattr(self, "tgi_container"):
            print("\t+ Stoping TGI container")
            self.tgi_container.stop()
            print("\t+ Waiting for TGI container to stop")
            self.tgi_container.wait()

        if hasattr(self, "docker_client"):
            print("\t+ Closing docker client")
            self.docker_client.close()
