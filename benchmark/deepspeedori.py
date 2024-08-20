
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch


params = {
    "top_k": 5,
    "max_new_tokens": 256,
    "penalty_alpha": 0.6
}

DEVICE = torch.device("cuda:0")
name = "mistralai/Mistral-7B-v0.3"
model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval().half()
tokenizer = AutoTokenizer.from_pretrained(name)

model = deepspeed.init_inference(
    model=model,     
    mp_size=1,       
    dtype=torch.float16, 
    replace_method="auto", 
    replace_with_kernel_inject=True,
)

prompt = "Quantum computers are"

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        **params
    )

print(prompt)
print()
print(tokenizer.decode(outputs[0])[len(prompt):].strip())
