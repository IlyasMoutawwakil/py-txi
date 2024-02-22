import numpy as np

from py_tgi import TEI, TGI

embed = TEI(model="bert-base-uncased", dtype="float16", pooling="mean", port=1234)
output = embed.encode("Hi, I'm a language model")
assert isinstance(output, np.ndarray)
output = embed.encode(["Hi, I'm a language model", "I'm fine, how are you?"])
assert isinstance(output, list) and all(isinstance(x, np.ndarray) for x in output)

llm = TGI(model="gpt2", sharded=False, port=4321)
output = llm.generate("Hi, I'm a sanity test")
assert isinstance(output, str)
output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
assert isinstance(output, list) and all(isinstance(x, str) for x in output)
