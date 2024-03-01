from py_txi.text_embedding_inference import TEI, TEIConfig
from py_txi.text_generation_inference import TGI, TGIConfig
from py_txi.utils import get_free_port

port = get_free_port()
ports = {"80/tcp": ("127.0.0.1", port)}

tei_config = TEIConfig(pooling="cls", ports=ports)
embed = TEI(tei_config)
output = embed.encode(["Hi, I'm an embedding model", "I'm fine, how are you?"])
print("Embed:", output)
embed.close()

port = get_free_port()
ports = {"80/tcp": ("127.0.0.1", port)}

tgi_config = TGIConfig(ports=ports)
llm = TGI(tgi_config)
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
print("LLM:", output)
llm.close()
