import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
from transformers import AutoTokenizer
from optimum.nvidia import AutoModelForCausalLM
import torch
import argparse

# Tricks to run inference faster based on Modern Nvidia GPUs
# debug, some profiler needs to be defined to test if the codes are running faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')


def load_model_and_tokenizer(model_name, 
                             use_fast, 
                             use_fp8, 
                             load_in_8bit, 
                             max_batch_size):
    """
    Some Args need to be explained a little bit:
    - use_fp8: use fp8 if you are using this gradio interface on Nvidia Hopper GPUs.
    - load_in_8bit: whether to use 8bit quantization.
    - max_batch_size: the max batch size for the model.
    """
    print('Loading model...')  
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.float16, 
                                                 device_map="auto",
                                                 use_fp8=use_fp8,
                                                 load_in_8bit=load_in_8bit,
                                                 max_batch_size=max_batch_size,
                                                 )
    return model, tokenizer


def inference(
            prompt,
            k, 
            alpha,
            model,
            tokenizer
            ):
    """
    Some Args need to be explained a little bit:
    - use_fp8: use fp8 
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading model into device...')

    print('Performing inference now for you, generating some texts...')
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(**model_inputs, penalty_alpha=alpha, top_k=k, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    one_generation_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return one_generation_text


def launch_gradio(fn):
    gr.Interface(fn=fn, 
                 inputs=[gr.inputs.Textbox(label="Prompt"),
                         gr.inputs.Radio(label="Dataset", choices=['book', 'wikinews', 'wikitext'], type="str"),
                         gr.inputs.Textbox(label="Model Name", type="str"),
                         gr.inputs.Slider(label="Top K", minimum=1, maximum=100, step=1, default=10),
                         gr.inputs.Slider(label="Penalty Alpha", minimum=0.0, maximum=2.0, step=0.1, default=0.0),
                         gr.inputs.Checkbox(label="Use FP8", default=False),
                         gr.inputs.Checkbox(label="Load in 8bit", default=False),
                         gr.inputs.Slider(label="Max Batch Size", minimum=1, maximum=8, step=1, default=8)
                        ],
                 outputs=gr.outputs.Textbox(label="Generated Text"),
                 title="NVIDIA Optimum Inference",
                 description="This is a gradio interface to perform inference with NVIDIA Optimum settings for GPT2 model."
                ).launch()


def main():
    if torch.cuda.is_available():
        print('Cuda is available.')
    else:
        print('Cuda is not available. Please check your CUDA configuration!')
        sys.exit(1)
    assert torch.cuda.device_count()==1, "This gradio script only supports single GPU. Please check your CUDA configuration!"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--use_fp8", type=bool, default=False)
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--max_batch_size", type=int, default=8)
    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.use_fp8, args.load_in_8bit, args.max_batch_size)

    def gradio_interface():
        return inference(model, tokenizer)
    
    launch_gradio(gradio_interface)

if __name__ == "__main__":
    main()
