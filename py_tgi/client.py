from huggingface_hub.inference._text_generation import TextGenerationResponse
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient
from typing import Any, Dict, List, Union
from logging import getLogger


LOGGER = getLogger("tgi-llm-client")

ClientOutput = Union[TextGenerationResponse, List[TextGenerationResponse]]


class BatchedInferenceClient:
    def __init__(self, url: str) -> None:
        LOGGER.info("\t+ Creating InferenceClient")
        self.tgi_client = InferenceClient(model=url)

    def generate(
        self, prompt: Union[str, List[str]], **kwargs: Dict[str, Any]
    ) -> ClientOutput:
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
