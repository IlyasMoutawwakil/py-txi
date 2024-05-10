import numpy as np

from py_txi import TEI, TGI, TEIConfig, TGIConfig


def test_cpu_tei():
    embed = TEI(config=TEIConfig(model_id="BAAI/bge-base-en-v1.5"))
    output = embed.encode("Hi, I'm a language model")
    assert isinstance(output, np.ndarray)
    output = embed.encode(["Hi, I'm a language model", "I'm fine, how are you?"])
    assert isinstance(output, list) and all(isinstance(x, np.ndarray) for x in output)
    embed.close()


def test_cpu_tgi():
    llm = TGI(config=TGIConfig(model_id="bigscience/bloom-560m"))
    output = llm.generate("Hi, I'm a sanity test")
    assert isinstance(output, str)
    output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
    assert isinstance(output, list) and all(isinstance(x, str) for x in output)
    llm.close()
