from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

if is_nvidia_system() or is_rocm_system():
    llm = TGI(
        model="TheBloke/Llama-2-7B-AWQ",  # awq model checkpoint
        devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,  # custom devices (ROCm)
        gpus="all" if is_nvidia_system() else None,  # all gpus (NVIDIA)
        quantize="gptq",  # use exllama kernels (rocm compatible)
    )
    output = llm.generate("Hi, I'm a sanity test")
    assert isinstance(output, str)
    output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
    assert isinstance(output, list) and all(isinstance(x, str) for x in output)

else:
    llm = TGI(model="gpt2", sharded=False)
    output = llm.generate("Hi, I'm a sanity test")
    assert isinstance(output, str)
    output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
    assert isinstance(output, list) and all(isinstance(x, str) for x in output)
