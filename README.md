# Py-TGI

Py-TGI is a Python wrapper around [TGI](https://github.com/huggingface/text-generation-inference) to enable creating and running TGI servers in a similar style to vLLM.

## Installation

```bash
pip install py-tgi
```

## Usage

Py-TGI is designed to be used in a similar way to vLLM. Here's an example of how to use it:

```python
from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

llm = TGI(
    model="TheBloke/Llama-2-7B-AWQ",  # awq model checkpoint
    devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,  # custom devices (ROCm)
    gpus="all" if is_nvidia_system() else None,  # all gpus (NVIDIA)
    quantize="gptq",  # use exllama kernels (rocm compatible)
)
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print(output)
```

Output: ```[" and I'm here to help you with any questions you have. What can I help you with", "\nUser 0: I'm doing well, thanks for asking. I'm just a"]```

That's it! Now you can write your Python scripts using the power of TGI.
