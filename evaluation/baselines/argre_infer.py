import json
import os
import sys
import logging
import inspect
import argparse
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
print(f"currentdir: {currentdir}, parentdir: {parentdir}, parentparentdir: {parentparentdir}")
from utils.startup import main
from utils.utils import setup_logging
from reward_model import MyLlamaForCausalLM, MyGPT2LMHeadModel, MyMistralForCausalLM, MyOPTForCausalLM

parser = argparse.ArgumentParser(description='ARGRE')
parser.add_argument('--config_file', type=str)
parser.add_argument('--score_model_path', type=str)
parser.add_argument('--hidden_dir', type=str, default='')
parser.add_argument('--guide_epochs', type=int, default=30)
parser.add_argument('--guide_lr', type=float, default=1)
parser.add_argument('--model_name_suffix', type=str, default='')
args = parser.parse_args()
config_filename = args.config_file
config = main(config_filename=config_filename)

model_name = config.get('Model', 'model_name')
logdir = os.path.join(os.environ["PROJECT_ROOT"], f'exp/ARGRE/{model_name}', model_name+'_'+args.model_name_suffix)
os.makedirs(logdir, exist_ok=True)
os.makedirs(os.path.join(logdir, "log"), exist_ok=True)
setup_logging(os.path.join(logdir, "log"), logname=f'run_{model_name}_epoch_{args.guide_epochs}_lr_{args.guide_lr}_{str(time.strftime("%Y-%m-%d-%H-%M-%S"))}', mode='w')
logging.info('Starting ARGRE')

import numpy as np
from utils.model_utils import load_large_model, MODEL_IDENITFIER
from utils.evaluate_model import evaluate_generate_ppl, evaluate_model_regressive

logging.info(f"args: {args}")
logging.info(f"guide_epochs: {args.guide_epochs}")
logging.info(f"guide_lr: {args.guide_lr}")


model_id = config.get('Model', 'model_name')

model_path = MODEL_IDENITFIER[model_id]
dtype = torch.float32 if model_id == 'gpt2' else torch.float16

def get_model_class(model_id):
    if 'gpt2' in model_id:
        return MyGPT2LMHeadModel
    elif 'llama' in model_id:
        return MyLlamaForCausalLM
    elif 'mistral' in model_id:
        return MyMistralForCausalLM
    elif 'opt' in model_id:
        return MyOPTForCausalLM
    else:
        raise ValueError(f'Invalid model_id: {model_id}')


model = get_model_class(model_id).from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        device_map="auto",
    )
model.set_score_model(args.score_model_path)


def load_data(save_dir, prefix="train"):
    hidden_states = torch.load(os.path.join(save_dir, f"{prefix}_hidden_states.pt"))
    attention_mask = torch.load(os.path.join(save_dir, f"{prefix}_attention_mask.pt"))
    with open(os.path.join(save_dir, f"{prefix}_record_data.json")) as f:
        record_data = json.load(f)
    return hidden_states, attention_mask, record_data

def load_direction(save_dir):
    direction = torch.load(os.path.join(save_dir, f"direction.pt"))
    return direction.squeeze(1)

print(f"Loading data from {args.hidden_dir}")
eval_hidden_states, eval_attention_mask, eval_record_data = load_data(args.hidden_dir, prefix='train')
direction = load_direction(args.hidden_dir)

def get_mean_reward(hidden_states, attention_mask, score_model, batch_size=512):
    
    score_model.eval()
    device = next(score_model.parameters()).device
    pos_hidden = hidden_states[:, 0] 
    neg_hidden = hidden_states[:, 1] 
    pos_mask = attention_mask[:, 0]   
    neg_mask = attention_mask[:, 1]   
    def compute_masked_reward(hidden, mask):
        flat_hidden = hidden.reshape(-1, hidden.size(-1)) 
        flat_mask = mask.reshape(-1).bool()  
        valid_hidden = flat_hidden[flat_mask] 
        num_valid = valid_hidden.size(0)
        
        rewards = []
        for i in tqdm(range(0, num_valid, batch_size), desc="Scoring tokens", disable=(num_valid <= batch_size)):
            batch = valid_hidden[i:i+batch_size].to(device)  
            batch_rewards = score_model(batch).squeeze(-1)  
            rewards.append(batch_rewards)
        return torch.cat(rewards)  

    pos_rewards = compute_masked_reward(pos_hidden, pos_mask)
    neg_rewards = compute_masked_reward(neg_hidden, neg_mask)
    
    pos_mean = pos_rewards.mean().cpu().item()
    neg_mean = neg_rewards.mean().cpu().item()
    return pos_mean, neg_mean

pos_mean, neg_mean = get_mean_reward(eval_hidden_states, eval_attention_mask, model.score)


model.set_lr_and_epochs(args.guide_lr, args.guide_epochs)
model.set_reward_direction(pos_mean, neg_mean, direction)

model.set_forward_mode("reward_guide")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path) 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


ppl, tox = -1, -1
ppl, tox, toxic_generate_list = evaluate_model_regressive(model, tokenizer)
logging.info(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}')


model.set_forward_mode("base")
gentox_ppl, toxic_generate_list = evaluate_generate_ppl(toxic_generate_list=toxic_generate_list, 
                    model=model, tokenizer=tokenizer)
logging.info(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}, Generate Perplexity: {gentox_ppl}')
    
save_file = os.path.join(logdir, f'{model_name}_epoch_{args.guide_epochs}_lr_{args.guide_lr}_toxic_generate_list.json')

with open(save_file, 'w') as f:
    json.dump(toxic_generate_list, f, indent=4)

with open(os.path.join(logdir, f'{model_name}_epoch_{args.guide_epochs}_lr_{args.guide_lr}.json'), 'w') as f:
    json.dump({
        "toxicity": tox*100,
        "perplexity": ppl,
        "generate_perplexity": gentox_ppl,
    }, f, indent=4)

