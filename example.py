from py_tgi import TGIServer, BatchedInferenceClient

tgi_server = TGIServer(model="gpt2", sharded=False)

try:
    client = BatchedInferenceClient(url=tgi_server.url)
    output = client.generate(["Hi, I'm an example 1", "Hi, I'm an example 2"])
    print("Output:", output)
except Exception as e:
    print(e)
finally:
    tgi_server.close()