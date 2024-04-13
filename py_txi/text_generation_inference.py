import asyncio
from dataclasses import dataclass
from logging import getLogger
from typing import Literal, Optional, Union

from .inference_server import InferenceServer, InferenceServerConfig
from .utils import is_nvidia_system, is_rocm_system

LOGGER = getLogger("Text-Generation-Inference")

Shareded_Literal = Literal["true", "false"]
DType_Literal = Literal["float32", "float16", "bfloat16"]
Quantize_Literal = Literal["bitsandbytes-nf4", "bitsandbytes-fp4", "gptq", "awq", "eetq", "fp8"]


@dataclass
class TGIConfig(InferenceServerConfig):
    # Launcher options
    num_shard: Optional[int] = None
    speculate: Optional[int] = None
    cuda_graphs: Optional[int] = None
    dtype: Optional[DType_Literal] = None
    sharded: Optional[Shareded_Literal] = None
    quantize: Optional[Quantize_Literal] = None
    disable_custom_kernels: Optional[bool] = None
    trust_remote_code: Optional[bool] = None
    # Concurrency options
    max_concurrent_requests: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.image is None:
            if is_nvidia_system() and self.gpus is not None:
                LOGGER.info("\t+ Using the latest NVIDIA GPU image for Text-Generation-Inference")
                self.image = "ghcr.io/huggingface/text-generation-inference:latest"
            elif is_rocm_system() and self.devices is not None:
                LOGGER.info("\t+ Using the latest ROCm AMD GPU image for Text-Generation-Inference")
                self.image = "ghcr.io/huggingface/text-generation-inference:latest-rocm"
            else:
                raise ValueError(
                    "Unsupported system. Please either provide the image to use explicitly "
                    "or use a supported system (NVIDIA/ROCm) while specifying gpus/devices."
                )

        if is_rocm_system() and "rocm" not in self.image:
            LOGGER.warning("\t+ You are running on a ROCm AMD GPU system but using a non-ROCM image.")


class TGI(InferenceServer):
    NAME: str = "Text-Generation-Inference"
    SUCCESS_SENTINEL: str = "Connected"
    FAILURE_SENTINEL: str = "Error"

    def __init__(self, config: TGIConfig) -> None:
        super().__init__(config)

    async def single_client_call(self, prompt: str, **kwargs) -> str:
        async with self.semaphore:
            output = await self.client.text_generation(prompt=prompt, **kwargs)
            return output

    async def batch_client_call(self, prompt: list, **kwargs) -> list:
        output = await asyncio.gather(*[self.single_client_call(prompt=p, **kwargs) for p in prompt])
        return output

    def generate(self, prompt: Union[str, list], **kwargs) -> Union[str, list]:
        if isinstance(prompt, str):
            output = asyncio.run(self.single_client_call(prompt, **kwargs))
            return output
        elif isinstance(prompt, list):
            output = asyncio.run(self.batch_client_call(prompt, **kwargs))
            return output
        else:
            raise ValueError(f"Unsupported input type: {type(prompt)}")
