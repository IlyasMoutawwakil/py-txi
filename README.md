# py-tgi

This repo constains two wrappers:

- `TGIServer`: a python wrapper around HuggingFace's TGI (text-generation-inference) using `docker-py`.
- `BatchedInferenceClient`: a python wrapper around HuggingFace's `InferenceClient` using threading to simulate batched inference.

Practical for running/managing TGI servers and benchmarking against other inference servers.

## Installation

```bash
python -m pip install git+https://github.com/IlyasMoutawwakil/py-tgi.git
```
