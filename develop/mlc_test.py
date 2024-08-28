### DEBUG 
### There is a very high chance that the beam search is not supported by mlc-llm at the moment.


from mlc_llm import MLCEngine
from mlc_llm.protocol.generation_config import GenerationConfig

model_name = "Mistral-7B-v0.3"
config = GenerationConfig(
    top_p = 0.9,
    temperature = 1.0,
)

chatmodule = MLCEngine(
    model = model_name,
)

for response in chatmodule._generate(
    prompt = "Hello, how are you?",
    generation_config = config,
    request_id = "test",
):
    print(response)
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)

print("\n")

chatmodule.terminate()
