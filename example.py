from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

if is_rocm_system():
    llm = TGI(
        model="TheBloke/Llama-2-7B-AWQ",  # model name
        image="ghcr.io/huggingface/text-generation-inference:latest-rocm",  # rocm image
        devices=["/dev/kfd", "/dev/dri"],  # all rocm devices
        quantize="gptq",  # use exllama kernels
    )
    output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
    print(output)

elif is_nvidia_system():
    llm = TGI(
        model="TheBloke/Llama-2-7B-AWQ",  # model name
        image="ghcr.io/huggingface/text-generation-inference:latest",  # rocm image
        quantize="gptq",  # use exllama kernels
        gpus="all",  # all gpus
    )
    output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
    print(output)

else:
    llm = TGI(
        model="gpt2",  # model name
        sharded=False,  # disable sharding on cpu
        image="ghcr.io/huggingface/text-generation-inference:latest-rocm",  # rocm image
    )
    output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
    print(output)
