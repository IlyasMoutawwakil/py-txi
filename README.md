# Py-TGI (Py-TXI at this point xD)

[![PyPI version](https://badge.fury.io/py/py-tgi.svg)](https://badge.fury.io/py/py-tgi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py-tgi)](https://pypi.org/project/py-tgi/)
[![PyPI - Format](https://img.shields.io/pypi/format/py-tgi)](https://pypi.org/project/py-tgi/)
[![Downloads](https://pepy.tech/badge/py-tgi)](https://pepy.tech/project/py-tgi)
[![PyPI - License](https://img.shields.io/pypi/l/py-tgi)](https://pypi.org/project/py-tgi/)
[![Tests](https://github.com/IlyasMoutawwakil/py-tgi/actions/workflows/test.yaml/badge.svg)](https://github.com/IlyasMoutawwakil/py-tgi/actions/workflows/test.yaml)

Py-TGI is a Python wrapper around [Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) and [Text-Embedding-Inference](https://github.com/huggingface/text-embeddings-inference) that enables creating and running TGI/TEI instances through the awesome `docker-py` in a similar style to Transformers API.

## Installation

```bash
pip install py-tgi
```

Py-TGI is designed to be used in a similar way to Transformers API. We use `docker-py` (instead of a dirty `subprocess` solution) so that the containers you run are linked to the main process and are stopped automatically when your code finishes or fails.

## Usage

Here's an example of how to use it:

```python
from py_tgi import TGI, is_nvidia_system, is_rocm_system

llm = TGI(
    model="NousResearch/Llama-2-7b-hf",
    devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,
    gpus="all" if is_nvidia_system() else None,
)
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print(output)
```

Output: ```[" and I'm here to help you with any questions you have. What can I help you with", "\nUser 0: I'm doing well, thanks for asking. I'm just a"]```

```python
from py_tgi import TEI

embed = TEI(
    model="BAAI/bge-large-en-v1.5",
    dtype="float16",
    pooling="mean",
    gpus="all" if is_nvidia_system() else None,
)
output = embed.encode(["Hi, I'm an embedding model", "I'm fine, how are you?"])
print(output)
```

Output: ```[array([[ 0.01058742, -0.01588806, -0.03487622, ..., -0.01613717,
         0.01772875, -0.02237891]], dtype=float32), array([[ 0.02815401, -0.02892136, -0.0536355 , ...,  0.01225784,
        -0.00241452, -0.02836569]], dtype=float32)]```

That's it! Now you can write your Python scripts using the power of TGI and TEI without having to worry about the underlying Docker containers.
