# Py-TGI

Py-TGI is a Python wrapper around [Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) that enables creating and running TGI instances through the awesome `docker-py` in a similar style to Transformers API.

## Installation

```bash
pip install py-tgi
```

## Usage

Py-TGI is designed to be used in a similar way to Transformers API. We use `docker-py` (instead of a dirty `subprocess` solution) so that the containers you run are linked to the main process and are stopped automatically when your code finishes or fails.
Here's an example of how to use it:

```python
from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

llm = TGI(
    model="TheBloke/Llama-2-7B-AWQ",  # awq model checkpoint
    quantize="gptq",  # use exllama kernels (awq compatible)
    devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,
    gpus="all" if is_nvidia_system() else None,
)
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print(output)
```

Output: ```[" and I'm here to help you with any questions you have. What can I help you with", "\nUser 0: I'm doing well, thanks for asking. I'm just a"]```

That's it! Now you can write your Python scripts using the power of TGI.
