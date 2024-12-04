""" Script for evaluation of LLMs for machine translation!

Requirements:
- pip install torch
- pip install transformers
- pip install bitsandbytes>=0.39.0 accelerate>=0.20.0

Sources:
- GPU HF Inference: https://huggingface.co/docs/transformers/perf_infer_gpu_one

"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

import argparse
from tqdm import tqdm
from pathlib import Path
from utils import config_utils

from data_tools.dataset_factory import get_original_dataset
from data_tools.prompt_tools import get_prompt
from evaluation.metrics import get_metric



QUANTIZATION_CONFIGS = {
    'fp4': BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4"),
    'np4': BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="np4"),
    'fp8': BitsAndBytesConfig(load_in_8bit=True),
}

MODEL_KWARGS_MAP = {
    'model': 'pretrained_model_name_or_path',
    'quantization': 'quantization_config',
    'device_map': 'device_map',
    'max_memory': 'max_memory',
}

TOKENIZER_KWARGS_MAP = {
    'tokenizer': 'pretrained_model_name_or_path',
    'batch_size': 'batch_size',
}

GENERATE_KWARGS_MAP = {
    'max_length': 'max_length',
    'num_return_sequences': 'num_return_sequences',
    'do_sample': 'do_sample',
    'top_k': 'top_k',
    'top_p': 'top_p',
}

def get_parser():
    parser = argparse.ArgumentParser(
        prog='MTEval',
        description='The program evaluates LLMs for machine translation task.',
        )
    parser.add_argument("--experiment",
                        help="Path to the experiment config YAML file.",
                        required=True,
                        type=str)
    parser.add_argument("--action",
                        help="Path to the experiment config YAML file.",
                        choices=['evaluation'],
                        required=False,
                        type=str)
    parser.add_argument("--model",
                        help="HF model name or model path.",
                        required=False,
                        type=str)
    parser.add_argument("--tokenizer",
                        help="HF model name or model path to acquire tokenizer.",
                        required=False,
                        type=str)
    parser.add_argument("--batch_size",
                        help="Set the batch size.",
                        required=False,
                        type=int)
    parser.add_argument("--max_length",
                        help="Set the maximal context length.",
                        required=False,
                        type=int)
    parser.add_argument("--device_map",
                        help="Set the memory allocation method.",
                        required=False,
                        type=str)
    parser.add_argument("--quantization",
                        help="Set the quantization method.",
                        required=False,
                        choices=QUANTIZATION_CONFIGS.keys(),
                        type=str)
    parser.add_argument("--prompts",
                        help="Path to the prompts config YAML file.",
                        required=False,
                        type=str)
    parser.add_argument("--instruct_lang",
                        help="Language code to set language for prompt instructions.",
                        required=False,
                        type=str)
    parser.add_argument("--start_lang",
                        help="Language code to set original language form which we translate.",
                        required=False,
                        type=str)
    parser.add_argument("--max_examples",
                        help="Set maximal number of examples per dataset to limit computation time.",
                        required=False,
                        type=int)
    parser.add_argument("--datasets",
                        help="List of dataset names.",
                        nargs='+',
                        required=False,
                        type=str)
    parser.add_argument("--metrics",
                        help="List of metrics.",
                        nargs='+',
                        required=False,
                        type=str)
    return parser
    

def map_kwargs(mapping: dict[str, str], **kwargs):
    _kwargs = {}
    for k in kwargs:
        if k in mapping:
            _kwargs[mapping[k]] = kwargs[k]
    return _kwargs


def get_model(**kwargs):
    _kwargs = map_kwargs(MODEL_KWARGS_MAP, **kwargs)
    # Set quantization
    if 'quantization_config' in _kwargs:
        _kwargs['quantization_config'] = QUANTIZATION_CONFIGS[_kwargs['quantization_config']]
    # Check if model path exists locally
    path = config_utils.get_model_path(_kwargs['pretrained_model_name_or_path'])
    if path.exists():
        _kwargs['pretrained_model_name_or_path'] = path
    try:
        return AutoModelForCausalLM.from_pretrained(**_kwargs)
    except:
        config_utils.huggingface_login()
        return AutoModelForCausalLM.from_pretrained(**_kwargs)


def get_tokenizer(**kwargs):
    _kwargs = map_kwargs(TOKENIZER_KWARGS_MAP, **kwargs)
    # Check if model path exists locally
    path = config_utils.get_model_path(_kwargs['pretrained_model_name_or_path'])
    if path.exists():
        _kwargs['pretrained_model_name_or_path'] = path
        return AutoTokenizer.from_pretrained(**_kwargs)
    return AutoTokenizer.from_pretrained(**_kwargs)


def get_dataset(dataset_name: str,
                prompts: str,
                start_lang: str,
                instruct_lang: str, 
                split: str='validation',
                randomize_prompts: bool=False):
    prompt = get_prompt(prompt_config=prompts,
                        instruct_lang=instruct_lang,
                        randomize_prompts=randomize_prompts)
    dataset = get_original_dataset(dataset_name)[split]
    dest_lang = list(set(dataset.column_names).difference(set([start_lang])))[0]
    def add_prompt(example):
        example[start_lang] = prompt.query_prompt(example[start_lang], start_lang)
        return example        
    dataset = dataset.map(add_prompt)
    dataset = dataset.rename_columns({start_lang: 'inputs', dest_lang: 'outputs'})
    return dataset


def get_datasets(datasets: list[str],
                 prompts: str,
                 start_lang: str,
                 instruct_lang: str, 
                 split: str='validation',
                 randomize_prompts: bool=False,
                 **kwargs):
    ans = {}
    for dataset_name in datasets:
        dataset = get_dataset(dataset_name, prompts, start_lang, instruct_lang, split, randomize_prompts)
        ans[dataset_name] = dataset
    return ans


def predict(**kwargs):
    """Get predictions for causal language model."""
    model = get_model(**kwargs)
    tokenizer = get_tokenizer(**kwargs)
    datasets = get_datasets(**kwargs)
    # Make and save predictions for each dataset        
    for dataset_name in datasets:
        predictions = []
        dataset = datasets[dataset_name]
        # Limit dataset to max_examples size
        if 'max_examples' in kwargs and kwargs['max_examples']>0:
            max_examples = min(kwargs['max_examples'], len(dataset))
        else:
            max_examples = len(dataset)
        # Generate text for the dataset
        for i in tqdm(range(max_examples)): 
            # Tokenize the input prompt
            prompt = dataset['inputs'][i]
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
            input_ids = input_ids.to(model.device)
            # Generate tokens using the model
            _kwargs = map_kwargs(GENERATE_KWARGS_MAP, **kwargs)
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("\n")] # TODO: terminators are not working
            print(model.device, input_ids.device)
            outputs = model.generate(input_ids, eos_token_id=terminators, **_kwargs)
            # Decode the generated tokens back to text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(generated_text[len(prompt):])
        # Save predictions for this dataset
        config_utils.save_experiment_output_prediction(inputs=dataset['inputs'],
                                                       outputs=dataset['outputs'],
                                                       predictions=predictions,
                                                       dataset_name=dataset_name,
                                                       experiment=kwargs['experiment'])


def evaluate_predictions(predictions: list[str],
                         references: list[str],
                         metrics: list[str]=["bleu"]) -> dict[str, float]:
    return {m: get_metric(m).compute(predictions=predictions, references=references) for m in metrics}


def evaluate_experiment_predictions(experiment_output_dir: Path=None) -> dict[str, list[str]]:
    """Evaluate predictions on all dataset in experiment output directory.

    Args:
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
    """
    kwargs = config_utils.get_experiment_output_config(experiment_output_dir=experiment_output_dir)
    for dataset_name in kwargs['datasets']:
        # Read dataset predictions
        data = config_utils.get_experiment_output_prediction(dataset_name, experiment_output_dir)
        scores = evaluate_predictions(data['predictions'], data['outputs'], kwargs['metrics'])
        config_utils.save_experiment_output_scores(scores, dataset_name=dataset_name, experiment=kwargs['experiment'])


def main(**kwargs):
    """Run evaluation experiment."""
    kwargs = config_utils.get_experiment_arguments(**kwargs)
    config_utils.save_experiment_output_config(kwargs, experiment=kwargs['experiment'])
    predict(**kwargs) # Make predictions and save them
    evaluate_experiment_predictions() # Evaluate predictions and save scores


if __name__=="__main__":
    main(get_parser().parse_args())
