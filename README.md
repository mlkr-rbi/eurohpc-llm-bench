# gemma2-challenge
Gemma2-challenge for kaggle competition

# environment setup
- run "generate_requirements.sh" to generate requirements.txt automatically, by scanning the code
- optionally, first edit generate_requirements.sh to add more packages 'by hand'
- run "pip install -r requirements.txt" to install all the required packages

- copy .gitignore.template to .gitignore and modify it as needed
- copy settings.template.py to settings.py and enter the values of the variables that will be used

- create new directories:
    - datasets - to hold datasets
    - models - to hold models
    - outputs - to hold outputs (for evaluation scores)

# code overview
- src/ contains the original training code by Filip and Miha
- data_tools contains tools and utils for creating and manipulating the datasets
- utils package is for various utility modules not located in more specific packages 
- experiments - contains yaml configurations for various experiments - create your own by copying existing ones and changing values
- prompts - contains yaml configurations for building prompts

## training
- training package contains the training code
- settings.MODEL_TRAINING_OUTPUT is the directory where the training output will be saved
- full_training_v1.py is a refactored and slightly updated version of DPmultiGPU.py
- full_training_v2.py is a version of full_training_v1.py that loads params from a .yaml file
- run_module.sh is a util script for server run that runs a module in the background and redirects the output

## evaluation
- metrics - contains functionality for evaluation metrics
- evaluation - contain the main functionality for evaluation of HF models

## running full_training_v2.py with run_module.sh and using a .yaml file (from the root directory)
- `./run_module.sh training/full_training_v2.py training/setup/train_setup_v1.yaml`

## running any action through main.py
### training
- `python main.py --experiment experiments/mt-train-2b-peft-3epochs_lr2e-4.yml`
- `CUDA_VISIBLE_DEVICES=0,1 python main.py --experiment experiments/mt-train-2b-peft-3epochs_lr2e-4.yml`
### evaluation
- `python main.py --experiment mt-eval-2b-it-v000.yml`
- `CUDA_VISIBLE_DEVICES=0,1 python main.py --experiment experiments/mt-eval-2b-peft-3epochs_lr2e-4-2024-12-01-v000.yml --max_examples 10`