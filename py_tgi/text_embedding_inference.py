import os
import time
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from huggingface_hub import InferenceClient

from .inference_server import InferenceServer
from .utils import CONNECTION_TIMEOUT, HF_CACHE_DIR

LOGGER = getLogger("TEI")


Pooling_Literal = Literal["cls", "mean"]
DType_Literal = Literal["float32", "float16"]


class TEI(InferenceServer):
    NAME: str = "Text-Embedding-Inference"

    def __init__(
        self,
        # model options
        model: str,
        revision: str = "main",
        # image options
        image: str = "ghcr.io/huggingface/text-embeddings-inference:latest",
        # docker options
        port: int = 1111,
        shm_size: str = "1g",
        address: str = "127.0.0.1",
        volumes: Dict[str, Any] = {HF_CACHE_DIR: "/data"},  # connects local hf cache to /data folder
        devices: Optional[List[str]] = None,  # e.g. ["/dev/kfd", "/dev/dri"] for ROCm
        gpus: Optional[Union[str, int]] = None,  # e.g. "all" or "0,1,2,3" or 4 for NVIDIA
        # launcher options
        # tgi launcher options
        dtype: Optional[DType_Literal] = None,
        pooling: Optional[Pooling_Literal] = None,
        tokenization_workers: Optional[int] = None,
        max_concurrent_requests: Optional[int] = None,
        max_batch_tokens: Optional[int] = None,
        max_batch_requests: Optional[int] = None,
        max_client_batch_size: Optional[int] = None,
    ) -> None:
        # tgi launcher options
        self.dtype = dtype
        self.pooling = pooling
        self.tokenization_workers = tokenization_workers
        self.max_concurrent_requests = max_concurrent_requests
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_requests = max_batch_requests
        self.max_client_batch_size = max_client_batch_size

        if gpus is None and "cpu-" not in image:
            LOGGER.warning("No GPUs were specified, but the image does not contain 'cpu-'. Adding it.")
            image_, tag_ = image.split(":")
            image = f"{image_}:cpu-{tag_}"

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
            if "Ready" in log:
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
                self.client.feature_extraction("Hello world!")
                LOGGER.info(f"\t+ Connected to {self.NAME} server successfully")
                return
            except Exception:
                LOGGER.info(f"\t+ {self.NAME} server is not ready yet, waiting 1 second")
                time.sleep(1)

        raise Exception(f"{self.NAME} server took too long to start (60 seconds)")

    def build_command(self):
        self.command = ["--model-id", self.model, "--revision", self.revision]
        if self.dtype:
            self.command.extend(["--dtype", self.dtype])
        if self.pooling:
            self.command.extend(["--pooling", self.pooling])
        if self.tokenization_workers:
            self.command.extend(["--tokenization-workers", str(self.tokenization_workers)])
        if self.max_concurrent_requests:
            self.command.extend(["--max-concurrent-requests", str(self.max_concurrent_requests)])
        if self.max_batch_tokens:
            self.command.extend(["--max-batch-tokens", str(self.max_batch_tokens)])
        if self.max_batch_requests:
            self.command.extend(["--max-batch-requests", str(self.max_batch_requests)])
        if self.max_client_batch_size:
            self.command.extend(["--max-client-batch-size", str(self.max_client_batch_size)])

    def build_env(self):
        self.env = {}
        if os.environ.get("HUGGING_FACE_HUB_TOKEN", None) is not None:
            self.env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def encode(self, text: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(text, str):
            output = self.client.feature_extraction(text=text, **kwargs)
            return output

        elif isinstance(text, list):
            outputs = []

            with ThreadPoolExecutor(max_workers=len(text)) as executor:
                futures = [
                    executor.submit(self.client.feature_extraction, text=text[i], **kwargs) for i in range(len(text))
                ]

            for i in range(len(text)):
                outputs.append(futures[i].result())

            return outputs

    def __call__(self, text: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        return self.encode(text, **kwargs)
