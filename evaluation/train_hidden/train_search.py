from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import time
os.chdir('../')


hidden_states_root="./ARGRE/evaluation/exp/collect_hidden_states"

def run_command(args: dict):
    print(f"Running task with args: {args}")
    command = [
        "python", "train_hidden/train_rm_hidden.py",
        "--config_file", f"{args['config_file']}",
        "--hidden_dir", f"{os.path.join(hidden_states_root, args['hidden_dir'])}",
        "--center_rewards", f"{str(args['center_rewards'])}",
        "--center_rewards_coefficient", str(args["center_rewards_coefficient"]),
        "--interpolation", str(args["interpolation"]),
        "--num_interpolations", str(args['num_interpolations']),
        "--train_epoch", str(args['train_epoch']),
        "--learning_rate", str(args['learning_rate']),
        "--dp_nums", str(args["dp_nums"]),
        "--model_name_suffix", f"{args['model_name_suffix']}",
    ]
    env = os.environ.copy() 
    env["CUDA_VISIBLE_DEVICES"] = args['cuda_id'] 
    try:
        subprocess.run(command, env=env)
        return f"Task completed successfully."
    except subprocess.CalledProcessError as e:
        return f"Task failed with error: {e}"

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
    dp_nums = [2000]
    config_file = configs[model_name]
    for dp_num in dp_nums:
        hidden_dir = f"{model_name}_toxicity_{dp_num}"

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for lr in learning_rate_list:
                for num_interpolations in num_interpolations_list:
                    if num_interpolations > 0:
                        interpolation = "True"
                    else:
                        interpolation = "False"
                
                    model_name_suffix = f"dp_{dp_num}_interp{num_interpolations}"
                    args = {
                        "cuda_id": "0,1",  
                        "config_file": f"{config_file}",
                        "hidden_dir": f"{hidden_dir}",  
                        "center_rewards": f"{True}", 
                        "center_rewards_coefficient": 0.01, 
                        "interpolation": f"{interpolation}",  
                        "num_interpolations": num_interpolations, 
                        "train_epoch": train_epoch, 
                        "learning_rate": lr,  
                        "dp_nums": dp_num,
                        "model_name_suffix": f"{model_name_suffix}", 
                    }
                    futures.append(executor.submit(run_command, args))

            for future in as_completed(futures):
                result = future.result()  
                print(result)

if __name__ == "__main__":
    
    run_model("llama-7b-hf")
