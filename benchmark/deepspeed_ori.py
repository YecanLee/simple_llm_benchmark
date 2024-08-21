import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch
import numpy as np
from datetime import datetime, timedelta
import time
from helper.text_utils import load_data, format_time, log_performance   

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument('--dataset_prefix', type=str, default='./data_ori')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--save_path_prefix', type=str, default='Mistralv03')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--alpha', required=True, type=float)
    parser.add_argument('--save_file', required=True, type=str)
    args = parser.parse_args()
    if torch.cuda.is_available():
        print('Cuda is available.') 
    cuda_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.cuda}' if cuda_available else 'cpu')

    assert args.dataset in ['book', 'wikinews', 'wikitext'], "Dataset must be one of 'book', 'wikinews', or 'wikitext'"
    full_data_path = f'{args.dataset_prefix}/{args.dataset}_contrastive_gpt2-xl_256.jsonl'
    print(f'Full data path is {full_data_path}')

    save_path_prefix = f'{args.save_path_prefix}/{args.dataset}/'
    print(f"Save path prefix is {save_path_prefix}")
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = f'{args.dataset}_{args.save_file}_k_{args.k}.json'
    save_path = os.path.join(save_path_prefix, save_name)
    print(f'Result saving path is {save_path}')

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="cpu")
    model = deepspeed.init_inference(
        model=model,     
        dtype=torch.float16,  
        replace_with_kernel_inject=True,
    )                   
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    model = torch.compile(model, mode="reduce-overhead")
    model.to(device)

    prefix_text_list, prefix_token_id_list, reference_text_list = load_data(full_data_path, tokenizer, mode=args.dataset)

    print('Performing inference...')
    data_num = len(prefix_text_list)
    print(data_num)
    result_list = []
    
    current_time = time.time()
    with torch.inference_mode():
        for index in range(20):
            print(f'Inference {index + 1}/{data_num} ({np.round((index + 1)/data_num*100, 2)} %)')
            one_prefix_text = prefix_text_list[index]
            one_reference_text = reference_text_list[index]
            model_inputs = tokenizer([one_prefix_text], return_tensors="pt").to(device)
            if 10 <= index < 20:
                start_time = time.time()
            generated_ids = model.generate(**model_inputs, 
                                           penalty_alpha=args.alpha, 
                                           top_k=args.k, 
                                           max_new_tokens=256, 
                                           pad_token_id=tokenizer.eos_token_id)
            one_generation_text = tokenizer.batch_decode(generated_ids)[0]
            if 10 <= index < 20:
                end_time = time.time()
                generation_time = end_time - start_time
                print(f"Generation time for prefix {index}: {format_time(int(generation_time))}")
        
        total_generation_time = sum([end_time - start_time for index in range(10, 20)])
        average_generation_time = total_generation_time / 10
        print(f"Average time for generation (prefix 10 to 20): {format_time(int(average_generation_time))}")
        approx_time = average_generation_time * data_num
        print("Approximate time for whole generation: ", format_time(int(approx_time)))

        now = datetime.now()
        estimated_finish_time = now + timedelta(seconds=int(approx_time))
        log_performance(args.model_name, 
                        args.dataset, 
                        args.k, 
                        args.alpha, 
                        average_generation_time, 
                        approx_time, 
                        save_path_prefix)
