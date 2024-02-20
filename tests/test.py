from py_tgi import TGI

llm = TGI("gpt2", gpus=None, devices=None, sharded=False)

output = llm.generate("Hi, I'm a sanity test")
assert isinstance(output, str)

output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
assert isinstance(output, list) and all(isinstance(x, str) for x in output)
