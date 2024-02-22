import os
import time
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Union

from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationResponse

from .inference_server import InferenceServer
from .utils import CONNECTION_TIMEOUT, HF_CACHE_DIR, is_rocm_system

LOGGER = getLogger("TGI")


DType_Literal = Literal["float32", "float16", "bfloat16"]
Quantize_Literal = Literal["bitsandbytes-nf4", "bitsandbytes-fp4", "gptq"]


class TGI(InferenceServer):
    NAME: str = "Text-Generation-Inference"

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
        volumes: Dict[str, Any] = {HF_CACHE_DIR: "/data"},  # connects local hf cache to /data folder
        devices: Optional[List[str]] = None,  # e.g. ["/dev/kfd", "/dev/dri"] for ROCm
        gpus: Optional[Union[str, int]] = None,  # e.g. "all" or "0,1,2,3" or 4 for NVIDIA
        # launcher options
        # tgi launcher options
        sharded: Optional[bool] = None,
        num_shard: Optional[int] = None,
        dtype: Optional[DType_Literal] = None,
        quantize: Optional[Quantize_Literal] = None,
        trust_remote_code: Optional[bool] = False,
        disable_custom_kernels: Optional[bool] = False,
    ) -> None:
        # tgi launcher options
        self.dtype = dtype
        self.sharded = sharded
        self.quantize = quantize
        self.num_shard = num_shard
        self.trust_remote_code = trust_remote_code
        self.disable_custom_kernels = disable_custom_kernels

        if devices and is_rocm_system() and "-rocm" not in image:
            LOGGER.warning("ROCm system detected, but the image does not contain '-rocm'. Adding it.")
            image = image + "-rocm"

        super().__init__(
            model=model,
            revision=revision,
            image=image,
            port=port,
            shm_size=shm_size,
            address=address,
            volumes=volumes,
            devices=devices,
            gpus=gpus,
        )

    def wait(self):
        for line in self.container.logs(stream=True):
            log = line.decode("utf-8").strip()
            if "Connected" in log:
                LOGGER.info(f"\t {log}")
                break
            elif "Error" in log:
                LOGGER.info(f"\t {log}")
                raise Exception(f"{self.NAME} server failed to start")
            else:
                LOGGER.info(f"\t {log}")

    def connect_client(self):
        start_time = time.time()
        while time.time() - start_time < CONNECTION_TIMEOUT:
            try:
                self.client = InferenceClient(model=self.url)
                self.client.text_generation("Hello world!")
                LOGGER.info(f"\t+ Connected to {self.NAME} server successfully")
                return
            except Exception:
                LOGGER.info(f"\t+ {self.NAME} server is not ready yet, waiting 1 second")
                time.sleep(1)

        raise Exception(f"{self.NAME} server took too long to start (60 seconds)")

    def build_command(self):
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

    def build_env(self):
        self.env = {}
        if os.environ.get("HUGGING_FACE_HUB_TOKEN", None) is not None:
            self.env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def generate(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Union[TextGenerationResponse, List[TextGenerationResponse]]:
        if isinstance(prompt, str):
            output = self.client.text_generation(prompt=prompt, **kwargs)
            return output

        elif isinstance(prompt, list):
            outputs = []

            with ThreadPoolExecutor(max_workers=len(prompt)) as executor:
                futures = [
                    executor.submit(self.client.text_generation, prompt=prompt[i], **kwargs) for i in range(len(prompt))
                ]

            for i in range(len(prompt)):
                outputs.append(futures[i].result())

            return outputs

    def __call__(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Union[TextGenerationResponse, List[TextGenerationResponse]]:
        return self.generate(prompt, **kwargs)
