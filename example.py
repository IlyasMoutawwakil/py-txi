from py_txi.text_embedding_inference import TEI, TEIConfig
from py_txi.text_generation_inference import TGI, TGIConfig

embed = TEI(config=TEIConfig(pooling="cls"))
output = embed.encode(["Hi, I'm an embedding model", "I'm fine, how are you?"] * 100)
print(len(output))
print("Embed:", output[0])
embed.close()

llm = TGI(config=TGIConfig(sharded="false"))
output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"] * 50)
print(len(output))
print("LLM:", output[0])
llm.close()
