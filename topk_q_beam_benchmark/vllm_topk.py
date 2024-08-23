import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm import LLM, SamplingParams
import torch
import numpy as np
from datetime import datetime, timedelta
import time 
from tqdm import trange
from helper.simple_prompts import load_data, format_time, log_performance

# run faster for new nvidia graphic cards
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
    parser.add_argument('--save_path_prefix', type=str, default='TimeBenchmark', help="save the logged inference benchmark performance")
    parser.add_argument('--cuda', type=int, default=0, help='cuda device id, for example, 0, 1, 2 ...')
    parser.add_argument('--save_file', required=True, type=str, help='save file name')
    parser.add_argument('--test_num', required=True, type=int, help='number of test samples')
    parser.add_argument('--sampling_methods', type=str, choices=['top_k', 'top_p', 'beam_search'], required=True, help='sampling methods')
    parser.add_argument('--top_k', type=int, default=5, help='top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p sampling parameter')
    parser.add_argument('--use_beam_search', type=bool, default=False, help='Whether to use beam search.')
    args = parser.parse_args()
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.cuda}' if cuda_available else 'cpu')

    assert args.dataset in ['book', 'wikinews', 'wikitext'], "Pre-defined dataset must be one of 'book', 'wikinews', or 'wikitext'"
    full_data_path = f'{args.dataset_prefix}/{args.dataset}_contrastive_gpt2-xl_256.jsonl'
    print(f'Full data path is {full_data_path}')

    save_path_prefix = f'{args.save_path_prefix}/{args.dataset}/'
    print(f"Save path prefix is {save_path_prefix}")
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = f'{args.dataset}_{args.save_file}_k_{args.k}.json'
    save_path = os.path.join(save_path_prefix, save_name)
    print(f'Result saving path is {save_path}')

    assert args.sampling_methods in ['top_k', 'top_p', 'beam_search'], "This benchmark only support top-k, top-p, \
    and beam search sampling methods. Please refer to other benchmark scripts for other sampling methods."
    if args.sampling_methods == 'top_k':
        sampling_params = SamplingParams(top_k=args.k)
    elif args.sampling_methods == 'top_p':
        sampling_params = SamplingParams(top_p=args.top_p)
    elif args.sampling_methods == 'beam_search':
        sampling_params = SamplingParams(top_k=args.k, num_beams=args.num_beams)

    print('Loading model by using the vllm library...')
    llm_model = LLM(args.model_name)

    prefix_text_list = load_data(full_data_path, mode=args.dataset)

    print('Performing inference now...')
    result_list = []

    current_time = time.time()
    with torch.inference_mode():
        for index in trange(args.test_num):
            one_prefix_text = prefix_text_list[index]
            if 10 <= index < 20:
                start_time = time.time()
            outputs = llm_model.generate(one_prefix_text, sampling_params)
            prompt = outputs.prompt
            generated_text = outputs.outputs[0].text    
            if 10 <= index < 20:
                end_time = time.time()
                generation_time = end_time - start_time
                print(f"Generation time for prefix {index}: {format_time(int(generation_time))}")
        
        total_generation_time = sum([end_time - start_time for index in range(10, 20)])
        average_generation_time = total_generation_time / 10
        print(f"Average time for generation (prefix 10 to 20): {format_time(int(average_generation_time))}")
        approx_time = average_generation_time * args.test_num
        print("Approximate time for whole generation: ", format_time(int(approx_time)))

        log_performance(save_path_prefix=save_path_prefix,  
                        model_name=args.model_name, 
                        dataset=args.dataset, 
                        k=args.k, 
                        alpha=args.alpha, 
                        average_generation_time=average_generation_time, 
                        approx_time=approx_time)
