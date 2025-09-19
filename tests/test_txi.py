import numpy as np

from py_txi import TEI, TGI, TEIConfig, TGIConfig


def test_cpu_tei():
    embed = TEI(config=TEIConfig(model_id="onnx-community/bge-base-en-v1.5-ONNX"))
    output = embed.encode("Hi, I'm a language model")
    assert isinstance(output, np.ndarray)
    output = embed.encode(["Hi, I'm a language model", "I'm fine, how are you?"])
    assert isinstance(output, list) and all(isinstance(x, np.ndarray) for x in output)
    embed.close()


def test_cpu_tgi():
    llm = TGI(config=TGIConfig(model_id="gpt2"))
    output = llm.generate("Hi, I'm a sanity test", max_new_tokens=2)
    assert isinstance(output, str)
    output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"], max_new_tokens=2)
    assert isinstance(output, list) and all(isinstance(x, str) for x in output)
    llm.close()
