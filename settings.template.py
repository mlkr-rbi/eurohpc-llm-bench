# copy to settings.py and fill in the values

import os
current_module_path = os.path.abspath(os.path.dirname(__file__))

### Change username with your name please - so we all know who produce these models
MODEL_TRAINING_OUTPUT = 'datasets/username'

HUGGINGFACE_TOKEN = ''

# tab-separated .txt file with parallel en-hr sentences
MACOCU_SENTENCES_FILE = 'datasets/MaCoCu-hr-en.sent.txt'
MACOCU_SENTENCES_FILE_SMALL = 'datasets/MaCoCu-hr-en.sent.1000.txt' # for development purposes

### Don't use MACOCU_DATASET_V1 - it is deprecated and will be removed!!!
MACOCU_DATASET_V1 = '' # pre-processed dataset, in huggingface dataset format, ready to use


DATASETS_DIR = "datasets" # Set relative path to store all datasets
MODELS_DIR = "models" # Set relative path to store all models
OUTPUTS_DIR = "outputs" # Set relative path to store outputs
PROMPTS_DIR = "prompts" # Set relative path to prompt configuration files 
EXPERIMENTS_DIR = "experiments" # Set relative path to experiments configuration files 

DEFAULT_EVALUATION_EXPERIMENT = "mt-eval-default.yml" # Set the name of the default evaluation experiment configuration file
DEFAULT_TRAINING_EXPERIMENT = "mt-train-default.yml" # Set the name of the default training experiment configuration file

MACOCU_PAIRS_DATASET_NAME = "macocu_train_dset_v1_pairs" # The name of the MACOCU HF dataset with sentence pairs.
MACOCU_TEXT_DATASET_NAME = "macocu_train_dset_v1_text" # The name of the MACOCU HF dataset with generated prompts, prepared for training.