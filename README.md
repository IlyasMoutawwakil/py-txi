# Py-TGI

Py-TGI is a Python wrapper around [TGI](https://github.com/huggingface/text-generation-inference) to enable creating and running TGI servers and clients in a similar style to vLLM

## Installation

```bash
pip install py-tgi
```

## Usage

Running a TGI server with a batched inference client:

```python
# from logging import basicConfig, INFO
# basicConfig(level=INFO)
from py_tgi import TGI

llm = TGI(model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ", quantize="awq")

try:
    output = llm.generate(["Hi, I'm an example 1", "Hi, I'm an example 2"])
    print("Output:", output)
except Exception as e:
    print(e)
finally:
    llm.close()
```

Output: [".\n\nHi, I'm an example 2.\n\nHi, I'm", ".\n\nI'm a simple example of a class that has a method that returns a value"]
