from py_tgi import TEI, TGI, is_nvidia_system, is_rocm_system

if is_rocm_system() or is_nvidia_system():
    llm = TGI(
        model="TheBloke/Llama-2-7B-AWQ",  # awq model checkpoint
        devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,  # custom devices (ROCm)
        gpus="all" if is_nvidia_system() else None,  # all gpus (NVIDIA)
        quantize="gptq",  # use exllama kernels (rocm compatible)
        port=4321,
    )
else:
    llm = TGI(model="gpt2", sharded=False)


output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print("LLM:", output)

if is_nvidia_system():
    embed = TEI(
        model="BAAI/bge-large-en-v1.5",
        dtype="float16",
        pooling="mean",
        gpus="all",
        port=1234,
    )
else:
    embed = TEI(model="BAAI/bge-large-en-v1.5", dtype="float16", pooling="mean")

output = embed.encode(["Hi, I'm an embedding model", "I'm fine, how are you?"])
print("Embed:", output)
