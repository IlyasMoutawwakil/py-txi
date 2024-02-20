# Py-TGI

Py-TGI is a Python wrapper around [TGI](https://github.com/huggingface/text-generation-inference) to enable creating and running TGI servers and clients in a similar style to vLLM

## Installation

```bash
pip install py-tgi
```

## Usage

Running a TGI server with a batched inference client:

```python
from logging import basicConfig, INFO
basicConfig(level=INFO) # to stream tgi container logs to stdout

from py_tgi import TGI

llm = TGI(model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ", quantize="awq")

try:
    output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
    print(output)
except Exception as e:
    print(e)
finally:
    # make sure to close the server
    llm.close()
```

Output: ```[" and I'm here to help you with any questions you have. What can I help you with", "\nUser 0: I'm doing well, thanks for asking. I'm just a"]```
