import json
import os
import sys
import logging
import inspect
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pca import PCA

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(f"currentdir: {currentdir}, parentdir: {parentdir}")

from utils.startup import main
from utils.utils import setup_logging
from utils.model_utils import MODEL_IDENITFIER

def make_parser():
    parser = argparse.ArgumentParser(description='collect_train')
    parser.add_argument('--config_file', default='gpt2-medium.ini', type=str, help='Config Filename. E.g. gpt2-medium.ini')
    parser.add_argument('--hidden_dir', type=str, default='')
    parser.add_argument('--model_name_suffix', type=str, default='')
    parser.add_argument('--rank', type=int, default=1)

    args = parser.parse_args()

    config_filename = args.config_file
    config = main(config_filename=config_filename)
    model_name = config.get('Model', 'model_name')
    logdir = os.path.join(os.environ["PROJECT_ROOT"], 'exp/collect_hidden_states', model_name+'_'+args.model_name_suffix)
    os.makedirs(logdir, exist_ok=True)
    setup_logging(logdir, logname=f'{model_name}_extract_direction')
    logging.info('Starting collect_train')
    return config, model_name, logdir, args

def load_data(save_dir, prefix="train"):
    hidden_states = torch.load(os.path.join(save_dir, f"{prefix}_hidden_states.pt"))
    attention_mask = torch.load(os.path.join(save_dir, f"{prefix}_attention_mask.pt"))
    with open(os.path.join(save_dir, f"{prefix}_record_data.json")) as f:
        record_data = json.load(f)
    return hidden_states, attention_mask, record_data



def get_last_valid_token(hidden_state, mask):
    valid_token_indices = torch.nonzero(mask == 1, as_tuple=False).squeeze()
    if valid_token_indices.dim() == 0:
        logging.info(valid_token_indices.shape)
        return None
    elif valid_token_indices.numel() > 0:
        logging.info(valid_token_indices.shape)
        last_valid_token_idx = valid_token_indices[-1].item()
    else:
        logging.info(f"error in get_last_valid_token, no valid token found")
        last_valid_token_idx = -1
        return None
    last_valid_token = hidden_state[last_valid_token_idx]
    logging.info(f"last_valid_token_idx: {last_valid_token_idx}")
    return last_valid_token

if __name__=="__main__":
    config, model_name, logdir, args = make_parser()
    train_hidden_states, train_attention_mask, train_record_data = load_data(args.hidden_dir, prefix='train')
    device = "cuda:0"
    sub_all = []
    for idx in range(0, len(train_hidden_states)): 
        hidden_states = train_hidden_states[idx] 
        attention_mask = train_attention_mask[idx] 
        
        pos_hidden_last = get_last_valid_token(hidden_states[0], attention_mask[0]) 
        neg_hidden_last = get_last_valid_token(hidden_states[1], attention_mask[1]) 
        if pos_hidden_last is None or neg_hidden_last is None:
            logging.info(f"deal the {idx}")
            continue
        sub_h = pos_hidden_last - neg_hidden_last
        sub_all.append(sub_h.unsqueeze(0)) 
    
    dim1, dim2 = sub_all[0].shape[0], sub_all[0].shape[1]
    llm_fit_data = torch.stack(sub_all, dim=0).to(device) 
    logging.info(f"llm_fit_data shape is {llm_fit_data.shape}")
    llm_pca = PCA(n_components=args.rank).to(device).fit(llm_fit_data.float())
    llm_direction = (llm_pca.components_ + llm_pca.mean_).mean(0).reshape(args.rank, dim1, dim2)
    logging.info(f"llm_direction shape is {llm_direction.shape}")

    torch.save(llm_direction.cpu(), os.path.join(logdir, f"direction.pt"))