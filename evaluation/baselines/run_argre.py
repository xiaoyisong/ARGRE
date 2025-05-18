from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import time
os.chdir('../')
total_cnt = 0

def run_command(args: dict, model_name):
    
    command = [
        "python", "baselines/argre_infer.py",
        "--config_file", f"{args['config_file']}",
        "--score_model_path", f"{args['score_model_path']}",
        "--guide_epochs", str(args["guide_epochs"]),
        "--guide_lr", str(args["guide_lr"]),
        "--model_name_suffix", str(args['model_name_suffix']),
        "--hidden_dir", f"{args['hidden_dir']}",
    ]
   
    env = os.environ.copy()  
    env["CUDA_VISIBLE_DEVICES"] = args['cuda_id']  
    
    try:
        subprocess.run(command, env=env)
        return f"Task with guide_epochs={args['guide_epochs']}, guide_lr={args['guide_lr']} completed successfully."
    except subprocess.CalledProcessError as e:
        return f"Task with guide_epochs={args['guide_epochs']}, guide_lr={args['guide_lr']} failed with error: {e}"

def run_tasks(tasks, num=1, model_name='gpt2'):

    with ThreadPoolExecutor(max_workers=num) as executor:
        futures = []
        for args in tasks:
            futures.append(executor.submit(run_command, args, model_name))

        for future in as_completed(futures):
            result = future.result()  
            print(result)


def args_each_score(score_model_path, model_name_suffix, config_file, hidden_dir):
    guide_epochs_list = [5]
    guide_lr_list = [0.5]
   
    futures = []
    cuda_ids = ["0,1"] 
    global total_cnt
    for guide_epochs in guide_epochs_list:
    
        for guide_lr in guide_lr_list:
            args = {
                "cuda_id": cuda_ids[total_cnt%len(cuda_ids)],
                "config_file": config_file, 
                "hidden_dir": hidden_dir,
                "score_model_path": score_model_path, 
                "guide_epochs": guide_epochs,
                "guide_lr": guide_lr,
                "model_name_suffix": model_name_suffix, 
            }
            futures.append(args)
            total_cnt += 1
    print(f"total is {total_cnt}")
    return futures

def run_model(model_name):
    configs = {
        "gpt2": "gpt2-medium.ini",
        "llama-7b-hf": "llama-7b.ini",
        "llama-13b-hf": "llama-13b.ini",
        "llama-30b-hf": "llama-30b.ini",
        "llama-7b-sft": "llama-7b-sft.ini",
        "mistral": "mistral-7b.ini",
        "mistral-sft": "mistral-sft-7b.ini",
        "opt": "opt-6.7b.ini"
    }
    num_interpolations_list = [7]
    learning_rate_list = [5e-4] 
    train_epoch = 3 
    dp_nums_list = [2000]

    config_file = configs[model_name]
    for dp_nums in dp_nums_list:
        
        hidden_states_root="./ARGRE/evaluation/exp/collect_hidden_states"
        model_dir = "./ARGRE/evaluation/exp/train_hidden_states"

        hidden_dir = f"{model_name}_toxicity_2000"
        hidden_dir = os.path.join(hidden_states_root, hidden_dir)
        tasks = []

        for num_interpolations in num_interpolations_list:
            model_name_suffix = f"dp_{dp_nums}_interp{num_interpolations}"
            traget_dir = os.path.join(model_dir, f'{model_name}_{model_name_suffix}')
            score_path = os.path.join(traget_dir, f"checkpoints/best_score_model.pth")

            if not os.path.exists(score_path):
                continue
            _task = args_each_score(score_path, model_name_suffix, config_file, hidden_dir)

            tasks.extend(_task)
        print(f"model: {model_name}, tasks: {len(tasks)}")
        run_tasks(tasks, num=1, model_name=model_name)

if __name__=="__main__":
    run_model("llama-7b-hf")