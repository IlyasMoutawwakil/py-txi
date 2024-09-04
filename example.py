from py_txi.text_embedding_inference import TEI, TEIConfig
from py_txi.text_generation_inference import TGI, TGIConfig

for gpus in [None, "1", "1,2"]:
    llm = TGI(config=TGIConfig(model_id="bigscience/bloom-560m", gpus=gpus))
    output = llm.generate(["Hi, I'm a language model", "I'm fine, how are you?"])
    print(len(output))
    print("LLM:", output)
    llm.close()

    embed = TEI(config=TEIConfig(model_id="BAAI/bge-base-en-v1.5", gpus=gpus))
    output = embed.encode(["Hi, I'm an embedding model", "I'm fine, how are you?"])
    print(len(output))
    print("Embed:", output)
    embed.close()
