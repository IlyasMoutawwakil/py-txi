# py-tgi

This repo constains two wrappers:

- `TGIServer`: a python wrapper around HuggingFace's TGI (text-generation-inference) using `docker-py`.
- `BatchedInferenceClient`: a python wrapper around HuggingFace's `InferenceClient` using threading to simulate batched inference.

Practical for running/managing TGI servers and benchmarking against other inference servers.

## Installation

```bash
python -m pip install git+https://github.com/IlyasMoutawwakil/py-tgi.git
```

## Usage

Running a TGI server with a batched inference client:

```python
from py_tgi import TGIServer, BatchedInferenceClient

tgi_server = TGIServer(model="gpt2", sharded=False)

try:
    client = BatchedInferenceClient(url=tgi_server.url)
    output = client.generate(["Hi, I'm an example 1", "Hi, I'm an example 2"])
    print("Output:", output)
except Exception as e:
    print(e)
finally:
    tgi_server.close()
```

Output:

```bash
INFO:tgi-server:        + Starting Docker client
INFO:tgi-server:        + Checking if TGI image exists
INFO:tgi-server:        + Building TGI command
INFO:tgi-server:        + Checking if GPU is available
INFO:tgi-server:        + Using GPU(s) from nvidia-smi
INFO:tgi-server:        + Using GPU(s): 0,1,2,4
INFO:tgi-server:        + Waiting for TGI server to be ready
INFO:tgi-server:         2024-01-15T08:47:15.882960Z  INFO text_generation_launcher: Args { model_id: "gpt2", revision: Some("main"), validation_workers: 2, sharded: Some(false), num_shard: None, quantize: None, speculate: None, dtype: None, trust_remote_code: false, max_concurrent_requests: 128, max_best_of: 2, max_stop_sequences: 4, max_top_n_tokens: 5, max_input_length: 1024, max_total_tokens: 2048, waiting_served_ratio: 1.2, max_batch_prefill_tokens: 4096, max_batch_total_tokens: None, max_waiting_tokens: 20, hostname: "ec83247f21ab", port: 80, shard_uds_path: "/tmp/text-generation-server", master_addr: "localhost", master_port: 29500, huggingface_hub_cache: Some("/data"), weights_cache_override: None, disable_custom_kernels: false, cuda_memory_fraction: 1.0, rope_scaling: None, rope_factor: None, json_output: false, otlp_endpoint: None, cors_allow_origin: [], watermark_gamma: None, watermark_delta: None, ngrok: false, ngrok_authtoken: None, ngrok_edge: None, env: false }
INFO:tgi-server:         2024-01-15T08:47:15.883089Z  INFO download: text_generation_launcher: Starting download process.
INFO:tgi-server:         2024-01-15T08:47:19.764449Z  INFO text_generation_launcher: Files are already present on the host. Skipping download.
INFO:tgi-server:         2024-01-15T08:47:20.387759Z  INFO download: text_generation_launcher: Successfully downloaded weights.
INFO:tgi-server:         2024-01-15T08:47:20.388064Z  INFO shard-manager: text_generation_launcher: Starting shard rank=0
INFO:tgi-server:         2024-01-15T08:47:26.062519Z  INFO text_generation_launcher: Server started at unix:///tmp/text-generation-server-0
INFO:tgi-server:         2024-01-15T08:47:26.095249Z  INFO shard-manager: text_generation_launcher: Shard ready in 5.70626412s rank=0
INFO:tgi-server:         2024-01-15T08:47:26.193466Z  INFO text_generation_launcher: Starting Webserver
INFO:tgi-server:         2024-01-15T08:47:26.204835Z  INFO hf_hub: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/hf-hub-0.3.2/src/lib.rs:55: Token file not found "/root/.cache/huggingface/token"
INFO:tgi-server:         2024-01-15T08:47:26.536395Z  INFO text_generation_router: router/src/main.rs:368: Serving revision 11c5a3d5811f50298f278a704980280950aedb10 of model gpt2
INFO:tgi-server:         2024-01-15T08:47:26.593914Z  INFO text_generation_router: router/src/main.rs:230: Warming up model
INFO:tgi-server:         2024-01-15T08:47:27.545238Z  WARN text_generation_router: router/src/main.rs:244: Model does not support automatic max batch total tokens
INFO:tgi-server:         2024-01-15T08:47:27.545255Z  INFO text_generation_router: router/src/main.rs:266: Setting max batch total tokens to 16000
INFO:tgi-server:        + Couldn't connect to TGI server
INFO:tgi-server:        + Retrying in 0.1s
INFO:tgi-server:        + TGI server ready at http://127.0.0.1:1111
INFO:tgi-llm-client:    + Creating InferenceClient
Output: [".0.0.0. I'm a programmer, I'm a programmer, I'm a", ".0.0. I'm a programmer, I'm a programmer, I'm a programmer,"]
INFO:tgi-server:        + Stoping TGI container
INFO:tgi-server:        + Waiting for TGI container to stop
INFO:tgi-server:        + Closing docker client
```
