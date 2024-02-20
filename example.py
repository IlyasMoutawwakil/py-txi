from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

llm = TGI(
    quantize="gptq",
    model="TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    gpus="0,1" if is_nvidia_system() else None,
    devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,
)
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print(output)
