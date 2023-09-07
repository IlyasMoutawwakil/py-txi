from py_tgi import TGIServer, LLMClient

server = TGIServer("gpt2", sharded=False)

try:
    client = LLMClient(url=server.url)
    output = client.generate("Hi, I'm a sanity test")
    print("Output:", output)
    server.close()
    assert isinstance(output, str)

# catch Exception and InterruptedError
except (Exception, InterruptedError, KeyboardInterrupt) as e:
    server.close()
    raise e
