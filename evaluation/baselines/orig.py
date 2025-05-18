import json
import os
import sys
import logging
import inspect
import argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.startup import main
from utils.utils import setup_logging

parser = argparse.ArgumentParser(description='Orig')
parser.add_argument('--config_file', default='gpt2-medium.ini', type=str, help='Config Filename. E.g. gpt2-medium.ini')

args = parser.parse_args()
config_filename = args.config_file
config = main(config_filename=config_filename)

model_name = config.get('Model', 'model_name')
logdir = os.path.join(os.environ["PROJECT_ROOT"], 'exp/orig', model_name)
print(f"logdir: {logdir}")
os.makedirs(logdir, exist_ok=True)
setup_logging(logdir)

from utils.model_utils import load_large_model
from utils.evaluate_model import evaluate_generate_ppl, evaluate_model


model_id = config.get('Model', 'model_name')
model, tokenizer = load_large_model(model_id)



logging.info('Evaluating perplexity and toxicity...')
ppl, tox, toxic_generate_list = evaluate_model(model, tokenizer)
logging.info(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}')
gentox_ppl, toxic_generate_list = evaluate_generate_ppl(toxic_generate_list=toxic_generate_list, 
                model=model, tokenizer=tokenizer)
logging.info(f'{model_id} - Perplexity: {ppl}, Toxicity: {tox}, Generate Perplexity: {gentox_ppl}')

save_file = os.path.join(logdir, f'{model_name}_toxic_generate_list.json')

with open(save_file, 'w') as f:
    json.dump(toxic_generate_list, f, indent=4)

with open(os.path.join(logdir, f'{model_name}.json'), 'w') as f:
    json.dump({
        "toxicity": tox*100,
        "perplexity": ppl,
        "generate_perplexity": gentox_ppl,
    }, f, indent=4)
