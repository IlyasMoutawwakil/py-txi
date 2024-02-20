# Py-TGI

Py-TGI is a Python wrapper around [TGI](https://github.com/huggingface/text-generation-inference) to enable creating and running TGI servers and clients in a similar style to vLLM

## Installation

```bash
pip install py-tgi
```

## Usage

Running a TGI server with a batched inference client:

```python
from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

llm = TGI(
    quantize="gptq",
    model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    gpus="0,1" if is_nvidia_system() else None,
    devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,
)
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print(output)

```

Output: ```[" and I'm here to help you with any questions you have. What can I help you with", "\nUser 0: I'm doing well, thanks for asking. I'm just a"]```
