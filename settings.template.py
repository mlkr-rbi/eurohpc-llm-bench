# copy to settings.py and fill in the values

import os
current_module_path = os.path.abspath(os.path.dirname(__file__))

MODEL_TRAINING_OUTPUT = ''

HUGGINGFACE_TOKEN = ''

# tab-separated .txt file with parallel en-hr sentences
MACOCU_SENTENCES_FILE = ''

MACOCU_DATASET_V1 = '' # pre-processed dataset, in huggingface dataset format, ready to use


DATASETS_DIR = "datasets" # Set relative path to store all datasets
MODELS_DIR = "models" # Set relative path to store all models
OUTPUTS_DIR = "outputs" # Set relative path to store outputs
PROMPTS_DIR = "prompts" # Set relative path to prompt configuration files 
EXPERIMENTS_DIR = "experiments" # Set relative path to experiments configuration files 
DEFAULT_EVALUATION_EXPERIMENT = "mt-eval-default.yml" # Set the name of the default evaluation experiment configuration file
DEFAULT_TRAINING_EXPERIMENT = "mt-train-default.yml" # Set the name of the default training experiment configuration file