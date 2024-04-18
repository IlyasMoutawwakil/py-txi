import asyncio
from dataclasses import dataclass
from logging import getLogger
from typing import Literal, Optional, Union

from .inference_server import InferenceServer, InferenceServerConfig
from .utils import is_rocm_system

LOGGER = getLogger("Text-Generation-Inference")

Shareded_Literal = Literal["true", "false"]
DType_Literal = Literal["float32", "float16", "bfloat16"]
Quantize_Literal = Literal["bitsandbytes-nf4", "bitsandbytes-fp4", "gptq"]


@dataclass
class TGIConfig(InferenceServerConfig):
    # Docker options
    image: str = "ghcr.io/huggingface/text-generation-inference:1.4.5-rocm"
    # Launcher options
    model_id: str = "gpt2"
    revision: str = "main"
    num_shard: Optional[int] = None
    dtype: Optional[DType_Literal] = None
    enable_cuda_graphs: Optional[bool] = None
    sharded: Optional[Shareded_Literal] = None
    quantize: Optional[Quantize_Literal] = None
    disable_custom_kernels: Optional[bool] = None
    trust_remote_code: Optional[bool] = None
    # Concurrency options
    max_concurrent_requests: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()

        if is_rocm_system() and "rocm" not in self.image:
            LOGGER.warning(
                "You are running on a ROCm system but the image is not rocm specific. "
                "Add 'rocm' to the image name to use the rocm specific image."
            )
            self.image += "-rocm"


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
