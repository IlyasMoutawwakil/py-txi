import asyncio
from dataclasses import dataclass
from logging import getLogger
from typing import List, Literal, Optional, Union

import numpy as np

from .inference_server import InferenceServer, InferenceServerConfig
from .utils import is_nvidia_system

LOGGER = getLogger("Text-Embedding-Inference")


Pooling_Literal = Literal["cls", "mean"]
DType_Literal = Literal["float32", "float16"]


@dataclass
class TEIConfig(InferenceServerConfig):
    # Launcher options
    dtype: Optional[DType_Literal] = None
    pooling: Optional[Pooling_Literal] = None
    # Concurrency options
    max_concurrent_requests: int = 512

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.image is None:
            if is_nvidia_system() and self.gpus is not None:
                LOGGER.info("\t+ Using the latest NVIDIA GPU image for Text-Embedding-Inference")
                self.image = "ghcr.io/huggingface/text-embeddings-inference:latest"
            else:
                LOGGER.info("\t+ Using the latest CPU image for Text-Embedding-Inference")
                self.image = "ghcr.io/huggingface/text-embeddings-inference:cpu-latest"

        if is_nvidia_system() and "cpu" in self.image:
            LOGGER.warning("\t+ You are running on a NVIDIA GPU system but using a CPU image.")

        if self.pooling is None:
            LOGGER.warning("\t+ Pooling strategy not provided. Defaulting to 'cls' pooling.")
            self.pooling = "cls"


class TEI(InferenceServer):
    NAME: str = "Text-Embedding-Inference"
    SUCCESS_SENTINEL: str = "Ready"
    FAILURE_SENTINEL: str = "Error"

    def __init__(self, config: TEIConfig) -> None:
        super().__init__(config)

    async def single_client_call(self, text: str, **kwargs) -> np.ndarray:
        async with self.semaphore:
            output = await self.client.feature_extraction(text=text, **kwargs)
            return output

    async def batch_client_call(self, text: List[str], **kwargs) -> List[np.ndarray]:
        output = await asyncio.gather(*[self.single_client_call(t, **kwargs) for t in text])
        return output

    def encode(self, text: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(text, str):
            output = asyncio.run(self.single_client_call(text, **kwargs))
            return output
        elif isinstance(text, list):
            output = asyncio.run(self.batch_client_call(text, **kwargs))
            return output
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")
