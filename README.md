### Code of ARGRE

#### Setup

- `pip install req.txt`
- Prepare the toxicity annotations and toxicity evaluation data in `./data`


#### Step 1: Identify the Non-toxic Direction

1. `cd ./ARGRE/evaluation/collect_hidden`
2. `bash collect.sh`, extracts hidden representations and saves them to `./exp/collect_hidden_states`  
3. `bash extract.sh`, computes the PCA direction and saves it to the same directory

#### Train the Autoregressive Reward Model (Modeling Toxicity Transition During Training)

1. `cd ./ARGRE/evaluation/train_hidden`
2. `python train_search.py`, trains the reward model and saves it to `./exp/train_hidden_states`

#### Detoxification during Inference

1. `cd ./ARGRE/evaluation/baselines`
2. `python run_argre.py`, runs ARGRE and saves evaluation results to `./exp/ARGRE`
