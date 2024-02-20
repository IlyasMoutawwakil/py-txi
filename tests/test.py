from py_tgi import TGI
from py_tgi.utils import is_nvidia_system, is_rocm_system

llm = TGI(
    "gpt2",
    sharded=False,
    gpus="0,1" if is_nvidia_system() else None,
    devices=["/dev/kfd", "/dev/dri"] if is_rocm_system() else None,
)

output = llm.generate("Hi, I'm a sanity test")
assert isinstance(output, str)

output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
assert isinstance(output, list) and all(isinstance(x, str) for x in output)
