import numpy as np

from py_txi.text_embedding_inference import TEI, TEIConfig
from py_txi.text_generation_inference import TGI, TGIConfig


def test_tei():
    embed = TEI(config=TEIConfig(pooling="cls"))
    output = embed.encode("Hi, I'm a language model")
    assert isinstance(output, np.ndarray)
    output = embed.encode(["Hi, I'm a language model", "I'm fine, how are you?"])
    assert isinstance(output, list) and all(isinstance(x, np.ndarray) for x in output)
    embed.close()


def test_tgi():
    llm = TGI(config=TGIConfig(dtype="float16"))
    output = llm.generate("Hi, I'm a sanity test")
    assert isinstance(output, str)
    output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
    assert isinstance(output, list) and all(isinstance(x, str) for x in output)
    llm.close()
