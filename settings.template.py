# copy to settings.py and fill in the values

import os
current_module_path = os.path.abspath(os.path.dirname(__file__))

MODEL_TRAINING_OUTPUT = ''

HUGGINGFACE_TOKEN = ''

# tab-separated .txt file with parallel en-hr sentences
MACOCU_SENTENCES_FILE = ''

MACOCU_DATASET_V1 = '' # pre-processed dataset, in huggingface dataset format, ready to use
