from logging import basicConfig, INFO
from py_tgi import TGI

basicConfig(level=INFO) # to stream tgi container logs

llm = TGI(model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ", quantization="awq")

try:
    output = llm.generate(["Hi, I'm an example 1", "Hi, I'm an example 2"])
    print("Output:", output)
except Exception as e:
    print(e)
finally:
    llm.close()
