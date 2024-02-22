from py_tgi import TEI, TGI, is_nvidia_system, is_rocm_system

if is_nvidia_system():
    llm = TGI(model="NousResearch/Llama-2-7b-hf", gpus="all", port=1234)
elif is_rocm_system():
    llm = TGI(model="NousResearch/Llama-2-7b-hf", devices=["/dev/kfd", "/dev/dri"], port=1234)
else:
    llm = TGI(model="NousResearch/Llama-2-7b-hf", port=1234)


output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print("LLM:", output)

if is_nvidia_system():
    embed = TEI(model="BAAI/bge-large-en-v1.5", dtype="float16", pooling="mean", gpus="all", port=4321)
else:
    embed = TEI(model="BAAI/bge-large-en-v1.5", dtype="float16", pooling="mean", port=4321)

output = embed.encode(["Hi, I'm an embedding model", "I'm fine, how are you?"])
print("Embed:", output)
