# copy to settings.py and fill in the values

import os
current_module_path = os.path.abspath(os.path.dirname(__file__))

### Change username with your name please - so we all know who produce these models
MODEL_TRAINING_OUTPUT = 'models/username'
TOKENIZER_CACHE_FOLDER = 'tokenizer_cache'

HUGGINGFACE_TOKEN = ''

DATASETS_DIR = "datasets" # Set relative path to store all datasets
MODELS_DIR = "models" # Set relative path to store all models
OUTPUTS_DIR = "outputs" # Set relative path to store outputs
PROMPTS_DIR = "prompts" # Set relative path to prompt configuration files
EXPERIMENTS_DIR = "experiments" # Set relative path to experiments configuration files

DEFAULT_EVALUATION_EXPERIMENT = "mt-eval-default.yml" # Set the name of the default evaluation experiment configuration file
DEFAULT_TRAINING_EXPERIMENT = "mt-train-default.yml" # Set the name of the default training experiment configuration file


# tab-separated .txt file with parallel en-hr sentences
MACOCU_SENTENCES_FILE = ''
MACOCU_SENTENCES_FILE_SMALL = '' # for development purposes
MACOCU_SENT_ORIG_HF = '' # full original macocu, in the HF dataset format

### Don't use MACOCU_DATASET_V1 - it is deprecated and will be removed!!!
MACOCU_DATASET_V1 = '' # pre-processed dataset, in huggingface dataset format, ready to use

MACOCU_PAIRS_DATASET_NAME = "macocu_train_dset_v1_pairs" # The name of the MACOCU HF dataset with sentence pairs.
MACOCU_TEXT_DATASET_NAME = "macocu_train_dset_v1_text" # The name of the MACOCU HF dataset with generated prompts, prepared for training.