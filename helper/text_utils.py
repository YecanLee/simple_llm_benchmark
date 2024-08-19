import json
import os
from datetime import datetime

def log_performance(model_name, dataset, k, alpha, average_time, total_time, save_path_prefix):
    log_data = {
        "model_name": model_name,
        "dataset": dataset,
        "k": k,
        "alpha": alpha,
        "average_time": average_time,
        "total_time": total_time,
        "timestamp": datetime.now().isoformat()
    }
    
    log_file = os.path.join(save_path_prefix, "performance_log.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_data) + "\n")

# parse text from a json file
def parse_text(item, tokenizer, mode):
    if mode == 'wikitext':
        prefix_text = item[0]['prompt'].strip(' ')
        full_text = item[0]['gold_ref'].strip(' ')
    else:
        prefix_text = item[0]['prompt']
        full_text = item[0]['gold_ref']
    prefix_token_list = tokenizer.tokenize(prefix_text)
    prefix_token_id_list = tokenizer.convert_tokens_to_ids(prefix_token_list)
    prefix_len = len(prefix_token_id_list)

    full_token_list = tokenizer.tokenize(full_text)
    full_token_id_list = tokenizer.convert_tokens_to_ids(full_token_list)
    reference_text = tokenizer.decode(full_token_id_list[prefix_len:])
    return prefix_text, prefix_token_id_list, reference_text

# load data from a json file
def load_data(in_f, tokenizer, mode):
    with open(in_f, 'r') as json_file:
        json_list = list(json_file)

    result_list = [json.loads(json_str) for json_str in json_list]
    
    prefix_text_list, prefix_token_id_list, reference_text_list = [], [], []
    for item in result_list:
        one_prefix_text, one_prefix_token_id, one_reference_text = parse_text(item, tokenizer, mode)
        prefix_text_list.append(one_prefix_text)
        prefix_token_id_list.append(one_prefix_token_id)
        reference_text_list.append(one_reference_text)
    return prefix_text_list, prefix_token_id_list, reference_text_list

# define a helper function to transform the time result to a human readable format
def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}h {minutes}m {seconds}s"
