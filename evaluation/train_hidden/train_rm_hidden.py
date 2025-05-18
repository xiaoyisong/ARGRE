import json
import logging
import os
import sys
import inspect
import time
import numpy as np
import torch
import argparse
from datasets import Dataset
from transformers import AutoConfig

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(f"currentdir: {currentdir}, parentdir: {parentdir}")#, parentparentdir: {parentparentdir}")

from rm_trainer import RMTrainer

import wandb

from utils.startup import main
from utils.utils import setup_logging
from utils.model_utils import MODEL_IDENITFIER

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_parser():
    parser = argparse.ArgumentParser(description='train_hidden')
    parser.add_argument('--config_file', default='gpt2-medium.ini', type=str, help='Config Filename. E.g. gpt2-medium.ini')
    parser.add_argument('--hidden_dir', type=str, default='')
    parser.add_argument('--model_name_suffix', type=str, default='')
    
    parser.add_argument('--interpolation', type=str2bool, default=False)
    parser.add_argument('--num_interpolations', type=int, default=0)
    ### 
    parser.add_argument('--center_rewards', type=str2bool, default=False)
    parser.add_argument('--center_rewards_coefficient', type=float, default=0.0)

    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--dp_nums', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--train_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--beta', type=float, default=0.05)

    args = parser.parse_args()

    config_filename = args.config_file
    config = main(config_filename=config_filename)
    model_name = config.get('Model', 'model_name')
    logdir = os.path.join(os.environ["PROJECT_ROOT"], f'exp/train_hidden_states/', model_name+'_'+args.model_name_suffix)
    os.makedirs(logdir, exist_ok=True)
    setup_logging(logdir, logname=f'{model_name}')
    logging.info('Starting collect_train')
    return config, model_name, logdir, args


def load_data(save_dir, prefix="train"):
    hidden_states = torch.load(os.path.join(save_dir, f"{prefix}_hidden_states.pt"))
    attention_mask = torch.load(os.path.join(save_dir, f"{prefix}_attention_mask.pt"))

    with open(os.path.join(save_dir, f"{prefix}_record_data.json")) as f:
        record_data = json.load(f)
    return hidden_states, attention_mask, record_data

def load_direction(save_dir):
    direction = torch.load(os.path.join(save_dir, f"direction.pt"))
    return direction.squeeze(1)

def load_model(config):
    model_id = config.get('Model', 'model_name')
    model_path = MODEL_IDENITFIER[model_id]
    dtype = torch.float32 if model_id == 'gpt2' else torch.float16
    model_config = AutoConfig.from_pretrained(model_path)
    logging.info(f"model_config: {model_config}")
    model, lm_head, tokenizer = None, None, None
    score_model = torch.nn.Sequential(
            torch.nn.Linear(model_config.hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)
        )
    del model
    return None, tokenizer, lm_head, score_model, dtype


class HiddenStateDataset(Dataset):
    def __init__(self, hidden_states, attention_masks, record_data):

        mask_sums = attention_masks.sum(dim=-1)  
        is_all_zero = (mask_sums == 0)       
        has_bad_pair = torch.any(is_all_zero, dim=1)  
        keep_indices = torch.where(~has_bad_pair)[0]  
        self.hidden_states = hidden_states[keep_indices]
        self.attention_masks = attention_masks[keep_indices]
        self.record_data = [record_data[i] for i in keep_indices.cpu().numpy()]  
        
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        hidden = self.hidden_states[idx]
        mask = self.attention_masks[idx]
        record = []
        for iid in idx:
            record.append(self.record_data[iid])
        
        return {
            'hidden_states': hidden,
            'attention_mask': mask,
            'record_data': record
        }

if __name__=="__main__":
    config, model_name, logdir, args = make_parser()
    _, tokenizer, lm_head, score_model, llm_dtype = load_model(config) 
    train_hidden_states, train_attention_mask, train_record_data = load_data(args.hidden_dir, prefix='train')
    if args.dp_nums:
        train_hidden_states = train_hidden_states[0:args.dp_nums]
        train_attention_mask = train_attention_mask[0:args.dp_nums]
        train_record_data = train_record_data[0:args.dp_nums]
    eval_hidden_states, eval_attention_mask, eval_record_data = load_data(args.hidden_dir, prefix='eval')
    if args.interpolation:
        direction = load_direction(args.hidden_dir)
        logging.info(f"direction shape is {direction.shape}")
    else:
        direction = None

    logging.info(f"train_hidden_states shape: {train_hidden_states.shape}, train_attention_mask shape: {train_attention_mask.shape}, train_record_data length: {len(train_record_data)}")
    logging.info(f"eval_hidden_states shape: {eval_hidden_states.shape}, eval_attention_mask shape: {eval_attention_mask.shape}, eval_record_data length: {len(eval_record_data)}")
    logging.info(f"data dtype is {train_hidden_states.dtype}")
    exp_name = str(time.strftime("%Y-%m-%d-%H-%M-%S"))
    wandb.init(project="training", name=exp_name, group=model_name)
    

    train_dataset = HiddenStateDataset(train_hidden_states, train_attention_mask, train_record_data)
    eval_dataset = HiddenStateDataset(eval_hidden_states, eval_attention_mask, eval_record_data)
    logging.info(f"train samples is {len(train_dataset)}")
    logging.info(f"eval samples is {len(eval_dataset)}")
    device = "cuda:0"
    trainer = RMTrainer(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        score_model=score_model,
        args=args,
        device=device,
        expdir=logdir,
        direction=direction,
    )

    trainer.train()

    wandb.finish()