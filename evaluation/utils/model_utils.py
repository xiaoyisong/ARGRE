import os
import torch
import logging
import numpy as np
from peft.peft_model import PeftModelForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.getLogger().setLevel(logging.INFO)


MODEL_IDENITFIER = {
    'gpt2': 'openai-community/gpt2-medium',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'mistral-sft': 'HuggingFaceH4/mistral-7b-sft-beta',
    'opt': 'facebook/opt-6.7b',
    'llama-7b-sft': "argsearch/llama-7b-sft-float32",
    'llama-7b-hf': "meta-llama/llama-7b-hf",
    'llama-13b-hf': "huggyllama/llama-13b",
    'llama-30b-hf': "huggyllama/llama-30b",
}


def load_large_model(model_id):
    model_path = MODEL_IDENITFIER[model_id]
    dtype = torch.float32 if model_id == 'gpt2' else torch.float16
    if model_id != "gpt2":
        dtype = torch.bfloat16 if model_id == 'llama-30b-hf' else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512,
                                              cache_dir=os.path.join(os.environ['HF_HOME'], 'hub'))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,  
        cache_dir=os.path.join(os.environ['HF_HOME'], 'hub'),
    )

    model.max_length = tokenizer.model_max_length
    model.eval()

    logging.info(f'Model {model_id} loaded.')
    return model, tokenizer


def get_model_category(model):
    """
    Returns the category of the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :return: Str
    """
    if isinstance(model, LlamaForCausalLM):
        return 'llama'
    if isinstance(model, MistralForCausalLM):
        return 'mistral'
    if isinstance(model, OPTForCausalLM):
        return 'opt'
    if isinstance(model, PeftModelForCausalLM):
        return get_model_category(model.model)
    if isinstance(model, GPT2LMHeadModel):
        return 'gpt2'

    raise ValueError('Unsupported model. Only GPT2 or LLaMA like architectures currently supported.')


def get_num_transformer_layers(model):
    """
    Returns the number of transformer layers in the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :return: Int
    """
    model_category = get_model_category(model)
    if model_category in ['llama', 'mistral']:
        return len(model.model.layers)          # 32
    elif model_category == 'opt':
        return len(model.model.decoder.layers)  # 32
    elif model_category in ['gpt2']:
        return len(model.transformer.h)         # 24 



def get_model_max_len(model):
    """
    Returns the maximum sequence length of the model.
    :param model: A language model subclassed by HuggingFace PreTrainedModel
    :return: Int
    """
    model_category = get_model_category(model)
    if model_category in ['gpt2']:
        return model.config.n_positions
    elif model_category in ['llama', 'mistral', 'opt']:
        return model.config.max_position_embeddings
