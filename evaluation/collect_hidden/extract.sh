cd ..
cuda=0,1

CUDA_VISIBLE_DEVICES=$cuda python collect_hidden/extract_direction.py --config_file llama-7b.ini \
    --model_name_suffix toxicity_2000 \
    --hidden_dir ./ARGRE/evaluation/exp/collect_hidden_states/llama-7b-hf_toxicity_2000
