import json
import os
import sys
import logging
import inspect
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
import numpy as np
from typing import Dict, Optional

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
print(f"currentdir: {currentdir}, parentdir: {parentdir}, parentparentdir: {parentparentdir}")
from utils.startup import main
from utils.utils import setup_logging

parser = argparse.ArgumentParser(description='collect_train')
parser.add_argument('--config_file', default='gpt2-medium.ini', type=str, help='Config Filename. E.g. gpt2-medium.ini')
parser.add_argument('--model_name_suffix', type=str, default='')
parser.add_argument('--data_nums', type=int, default=2000)
args = parser.parse_args()
config_filename = args.config_file
config = main(config_filename=config_filename)

model_name = config.get('Model', 'model_name')
logdir = os.path.join(os.environ["PROJECT_ROOT"], 'exp/collect_hidden_states', model_name+'_'+args.model_name_suffix+'_'+str(args.data_nums))
os.makedirs(logdir, exist_ok=True)
setup_logging(logdir, logname=f'{model_name}')
logging.info('Starting collect_train')

from utils.model_utils import MODEL_IDENITFIER



def get_toxic_datset(
    split: str = "train",
    sanity_check: bool = False,
    num_proc=4,
) -> Dataset:
    
    filedir = "./ARGRE/data/toxicity_pairwise"
    if split == "train":
        filepath = os.path.join(filedir, 'split_0.jsonl')
        default_num_dps = args.data_nums
    else:
        filepath = os.path.join(filedir, 'split_1.jsonl')
        default_num_dps = 500
    def split_common_prefix(str1, str2):
        common_prefix = ""
        for ch1, ch2 in zip(str1, str2):
            if ch1 == ch2:
                common_prefix += ch1
            else:
                break

        continuation1 = str1[len(common_prefix):]
        continuation2 = str2[len(common_prefix):]
        return common_prefix, continuation1, continuation2
    
    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            json_object = json.loads(line)

            prompt, cont_good, cont_bad = split_common_prefix(json_object['unpert_gen_text'], json_object['pert_gen_text'])
            dp = {
                "prompt": prompt,
                "chosen": cont_good,
                "rejected": cont_bad,
            }
            data_list.append(dp)

    dataset = Dataset.from_list(data_list)
    
    if split == "train":
        indices = np.random.choice(len(data_list), default_num_dps, replace=False)
        data_list_sample = [data_list[i] for i in indices]
        logging.info(f"{indices}")
        dataset = Dataset.from_list(data_list_sample)
    else:   
        dataset = dataset.select(range(0, default_num_dps))
        
    return dataset

model_id = config.get('Model', 'model_name')

model_path = MODEL_IDENITFIER[model_id]
dtype = torch.float32 if model_id == 'gpt2' else torch.float16



model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto",
    )
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=1024)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id




    
train_dataset = get_toxic_datset(split="train")
eval_dataset = get_toxic_datset(split="test")


def eval_model(dataset, model, tokenizer):
    record_data = []
    id = 0
    hidden_states_all = []
    attention_mask_all = []
    max_length = 0
    for data in tqdm(dataset, total=len(dataset)):
        id = id + 1
        prompt_data = [
            data['prompt'] + data['chosen'], 
            data['prompt'] + data['rejected'], 
        ]
        inputs = tokenizer(prompt_data, return_tensors="pt", padding=True, truncation=True)
        prompt_ids = tokenizer(data['prompt'], return_tensors="pt")
        
        start_index = prompt_ids.input_ids.shape[1] - 1
        if id == 1:
            print(f"start_index: {start_index}")
            print(f"prompt_ids: {prompt_ids.input_ids}")
            print(f"inputs[0]: {inputs.input_ids[0]}")
            print(f"inputs[1]: {inputs.input_ids[1]}")
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
        
        current_length = inputs.attention_mask.size(1)
        if current_length > max_length:
            max_length = current_length
        
        hidden_states_all.append(outputs.hidden_states[-1])
        attention_mask_all.append(inputs.attention_mask)

        output_data = {
            "id": id,
            "chosen": prompt_data[0],
            "reject": prompt_data[1],
            "start_index": start_index,
        }
        record_data.append(output_data)


    batch_size = len(hidden_states_all)
    hidden_size = hidden_states_all[0].size(-1)
    

    padded_hidden_states = torch.zeros(
        (batch_size, 2, max_length, hidden_size),
        dtype=hidden_states_all[0].dtype
    )
    padded_attention_mask = torch.zeros(
        (batch_size, 2, max_length),
        dtype=attention_mask_all[0].dtype
    )
    
    for i, (hs, am) in enumerate(zip(hidden_states_all, attention_mask_all)):
        seq_len = hs.size(1)
        padded_hidden_states[i, :, :seq_len, :] = hs
        padded_attention_mask[i, :, :seq_len] = am 
    return padded_hidden_states, padded_attention_mask, record_data


def save_hidden_states(hidden_states, attention_mask, record_data, output_dir, prefix="train"):
    print(f"save {prefix}")
    print(f"hidden_states.shape: {hidden_states.shape}")
    print(f"attention_mask.shape: {attention_mask.shape}")
    torch.save(hidden_states, os.path.join(output_dir, f"{prefix}_hidden_states.pt"))
    torch.save(attention_mask, os.path.join(output_dir, f"{prefix}_attention_mask.pt"))
    with open(os.path.join(output_dir, f"{prefix}_record_data.json"), 'w') as f:
        json.dump(record_data, f, ensure_ascii=False, indent=4)

output_dir = logdir

train_hidden_states, train_attention_mask, train_record_data = eval_model(train_dataset, model, tokenizer)
save_hidden_states(train_hidden_states, train_attention_mask, train_record_data, output_dir, prefix="train")
eval_hidden_states, eval_attention_mask, eval_record_data = eval_model(eval_dataset, model, tokenizer)
save_hidden_states(eval_hidden_states, eval_attention_mask, eval_record_data, output_dir, prefix="eval")

