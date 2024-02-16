# py-tgi

This presents a Python wrapper for the using `docker-py` to manage a TGI server and a client for single and batched inference.

## Installation

```bash
python -m pip install git+https://github.com/IlyasMoutawwakil/py-tgi.git
```

## Usage

Running a TGI server with a batched inference client:

```python
# from logging import basicConfig, INFO
# basicConfig(level=INFO)
from py_tgi import TGI

llm = TGI(model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ", quantization="awq")

try:
    output = llm.generate(["Hi, I'm an example 1", "Hi, I'm an example 2"])
    print("Output:", output)
except Exception as e:
    print(e)
finally:
    llm.close()
```

Output:

```bash
Output: [".\n\nHi, I'm an example 2.\n\nHi, I'm", ".\n\nI'm a simple example of a class that has a method that returns a value"]
```
