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
    # Docker options
    image: str = "ghcr.io/huggingface/text-embeddings-inference:cpu-latest"
    # Launcher options
    model_id: str = "bert-base-uncased"
    revision: str = "main"
    dtype: Optional[DType_Literal] = None
    pooling: Optional[Pooling_Literal] = None
    # Concurrency options
    max_concurrent_requests: int = 512

    def __post_init__(self) -> None:
        super().__post_init__()

        if is_nvidia_system() and "cpu" in self.image:
            LOGGER.warning(
                "Your system has NVIDIA GPU, but you are using a CPU image."
                "Consider using a GPU image for better performance."
            )


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
