cd ..
cuda=0,1

CUDA_VISIBLE_DEVICES=$cuda python collect_hidden/collect_hidden.py --config_file llama-7b.ini \
    --model_name_suffix toxicity

