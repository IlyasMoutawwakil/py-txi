from py_tgi import TGIModel

gpt2 = TGIModel("gpt2")

assert isinstance(gpt2.generate("Hello world!"), str)