import asyncio
from dataclasses import dataclass
from logging import getLogger
from typing import Literal, Optional, Union

from .docker_inference_server import DockerInferenceServer, DockerInferenceServerConfig

LOGGER = getLogger("TGI")

Shareded_Literal = Literal["true", "false"]
DType_Literal = Literal["float32", "float16", "bfloat16"]
Quantize_Literal = Literal["bitsandbytes-nf4", "bitsandbytes-fp4", "gptq"]


@dataclass
class TGIConfig(DockerInferenceServerConfig):
    # Docker options
    image: str = "ghcr.io/huggingface/text-generation-inference:latest"
    # Launcher options
    model_id: str = "gpt2"
    revision: str = "main"
    dtype: Optional[DType_Literal] = None
    quantize: Optional[Quantize_Literal] = None
    sharded: Optional[Shareded_Literal] = None
    num_shard: Optional[int] = None
    trust_remote_code: Optional[bool] = None
    disable_custom_kernels: Optional[bool] = None
    # Inference options
    max_best_of: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    max_stop_sequences: Optional[int] = None
    max_top_n_tokens: Optional[int] = None
    max_input_length: Optional[int] = None
    max_total_tokens: Optional[int] = None
    waiting_served_ratio: Optional[float] = None
    max_batch_prefill_tokens: Optional[int] = None
    max_batch_total_tokens: Optional[int] = None
    max_waiting_tokens: Optional[int] = None
    max_batch_size: Optional[int] = None
    enable_cuda_graphs: Optional[bool] = None
    huggingface_hub_cache: Optional[str] = None
    weights_cache_override: Optional[str] = None
    cuda_memory_fraction: Optional[float] = None
    rope_scaling: Optional[str] = None
    rope_factor: Optional[str] = None
    json_output: Optional[bool] = None
    otlp_endpoint: Optional[str] = None
    cors_allow_origin: Optional[list] = None
    watermark_gamma: Optional[str] = None
    watermark_delta: Optional[str] = None
    tokenizer_config_path: Optional[str] = None
    disable_grammar_support: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__post_init__()


class TGI(DockerInferenceServer):
    NAME: str = "Text-Generation-Inference"
    SUCCESS_SENTINEL: str = "Connected"
    FAILURE_SENTINEL: str = "Error"

    def __init__(self, config: TGIConfig) -> None:
        super().__init__(config)

    async def single_client_call(self, prompt: str, **kwargs) -> str:
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
