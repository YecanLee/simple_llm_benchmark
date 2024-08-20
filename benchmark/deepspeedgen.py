from mii import pipeline
pipe = pipeline("mistralai/Mistral-7B-v0.3")
output = pipe(["Hello, my name is", "DeepSpeed is"], top_k=5, max_new_tokens=256, penalty_alpha=0.6)
print(output)