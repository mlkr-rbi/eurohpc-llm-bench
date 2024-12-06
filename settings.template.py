# copy to settings.py and fill in the values

import os
current_module_path = os.path.abspath(os.path.dirname(__file__))

### Using MODEL_TRAINING_OUTPUT will soon be deprecated!!!
### In the future all models will be placed within directory DATASETS_DIR!
MODEL_TRAINING_OUTPUT = ''

HUGGINGFACE_TOKEN = ''

# tab-separated .txt file with parallel en-hr sentences
MACOCU_SENTENCES_FILE = ''

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