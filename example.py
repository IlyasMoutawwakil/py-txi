from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

if is_rocm_system() or is_nvidia_system():
    llm = TGI(
        model="TheBloke/Llama-2-7B-AWQ",  # awq model checkpoint
        devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,  # custom devices (ROCm)
        gpus="all" if is_nvidia_system() else None,  # all gpus (NVIDIA)
        quantize="gptq",  # use exllama kernels (rocm compatible)
    )
else:
    llm = TGI(model="gpt2", sharded=False)

output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print(output)
