from logging import basicConfig, INFO
from py_tgi import TGI

basicConfig(level=INFO)

llm = TGI("gpt2", sharded=False)

try:
    output = llm.generate("Hi, I'm a sanity test")
    assert isinstance(output, str)

    output = llm.generate(["Hi, I'm a sanity test", "I'm a second sentence"])
    assert isinstance(output, list)

    llm.close()

# catch Exception and InterruptedError
except (Exception, InterruptedError, KeyboardInterrupt) as e:
    llm.close()
    raise e
