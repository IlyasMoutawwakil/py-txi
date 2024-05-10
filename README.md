# Py-TXI

[![PyPI version](https://badge.fury.io/py/py-txi.svg)](https://badge.fury.io/py/py-txi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py-txi)](https://pypi.org/project/py-txi/)
[![PyPI - Format](https://img.shields.io/pypi/format/py-txi)](https://pypi.org/project/py-txi/)
[![Downloads](https://pepy.tech/badge/py-txi)](https://pepy.tech/project/py-txi)
[![PyPI - License](https://img.shields.io/pypi/l/py-txi)](https://pypi.org/project/py-txi/)
[![Test](https://github.com/IlyasMoutawwakil/py-txi/actions/workflows/test.yaml/badge.svg)](https://github.com/IlyasMoutawwakil/py-txi/actions/workflows/tests.yaml)

Py-TXI is a Python wrapper around [Text-Generation-Inference](https://github.com/huggingface/text-generation-inference) and [Text-Embedding-Inference](https://github.com/huggingface/text-embeddings-inference) that enables creating and running TGI/TEI instances through the awesome `docker-py` in a similar style to Transformers API.

## Installation

```bash
pip install py-txi
```

Py-TXI is designed to be used in a similar way to Transformers API. We use `docker-py` (instead of a dirty `subprocess` solution) so that the containers you run are linked to the main process and are stopped automatically when your code finishes or fails.

## Advantages

- **Easy to use**: Py-TXI is designed to be used in a similar way to Transformers API.
- **Automatic cleanup**: Py-TXI stops the Docker container when your code finishes or fails.
- **Batched inference**: Py-TXI supports sending a batch of inputs to the server for inference.
- **Automatic port allocation**: Py-TXI automatically allocates a free port for the Inference server.
- **Configurable**: Py-TXI allows you to configure the Inference servers using a simple configuration object.
- **Verbose**: Py-TXI streams the logs of the underlying Docker container to the main process so you can debug easily.

## Usage

Here's an example of how to use it:

```python
from py_txi import TGI, TGIConfig

llm = TGI(config=TGIConfig(model_id="bigscience/bloom-560m", gpus="0"))
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print("LLM:", output)
llm.close()
```

Output: ```LLM: [' student. I have a problem with the following code. I have a class that has a method that', '"\n\n"I\'m fine," said the girl, "but I don\'t want to be alone.']```

```python
from py_txi import TEI, TEIConfig

embed = TEI(config=TEIConfig(model_id="BAAI/bge-base-en-v1.5"))
output = embed.encode(["Hi, I'm an embedding model", "I'm fine, how are you?"])
print("Embed:", output)
embed.close()
```

Output: ```[array([[ 0.01058742, -0.01588806, -0.03487622, ..., -0.01613717,
         0.01772875, -0.02237891]], dtype=float32), array([[ 0.02815401, -0.02892136, -0.0536355 , ...,  0.01225784,
        -0.00241452, -0.02836569]], dtype=float32)]```

That's it! Now you can write your Python scripts using the power of TGI and TEI without having to worry about the underlying Docker containers.
