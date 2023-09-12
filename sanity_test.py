from py_tgi import TGIServer, BatchedInferenceClient

tgi_server = TGIServer("gpt2", sharded=False)

try:
    client = BatchedInferenceClient(url=tgi_server.url)
    output = client.generate("Hi, I'm a sanity test")
    print("Output:", output)
    tgi_server.close()
    assert isinstance(output, str)

# catch Exception and InterruptedError
except (Exception, InterruptedError, KeyboardInterrupt) as e:
    tgi_server.close()
    raise e
