''' training in torch, basically a more structured version of DPmultiGPU.py code '''
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
import yaml
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

from data_tools.dataset_factory import get_macocu_text_v1, get_test_cro_dataset
from data_tools.dataset_utils import discard_columns
from utils import config_utils
from utils.config_utils import get_tokenizer_cache_folder
from utils.hf_utils import remove_hf_checkpoints


def get_parser():
    parser = argparse.ArgumentParser(
        prog='cro-gemma',
        description='The program for training and evaluating LLMs for Croatian language.',
        )
    parser.add_argument("--experiment",
                        help="Path to the experiment config YAML file.",
                        required=True,
                        type=str)
    parser.add_argument("--action",
                        help="Path to the experiment config YAML file.",
                        choices=['training'],
                        required=False,
                        type=str)
    parser.add_argument("--model_id",
                        help="HF model name or model path.",
                        required=False,
                        type=str)
    parser.add_argument("--dataset_label",
                        help="The label of the dataset.",
                        required=False,
                        type=str)
    parser.add_argument("--gradient_accumulation_steps",
                        help="Set the batch size.",
                        required=False,
                        type=int)
    parser.add_argument("--per_device_train_batch_size",
                        help="Set the batch size.",
                        required=False,
                        type=int)
    parser.add_argument("--per_device_eval_batch_size",
                        help="Set the batch size.",
                        required=False,
                        type=int)
    parser.add_argument("--max_seq_length",
                        help="Set the maximal context length.",
                        required=False,
                        type=int)
    parser.add_argument("--deepspeed",
                        help="Path to the deepspeed config .json file.",
                        required=False,
                        type=str)
    # cached_tokenization, 'tokenize_only'
    parser.add_argument("--cached_tokenization",
                        help="Use cached tokenization.",
                        action='store_true')
    parser.add_argument("--tokenize_only",
                        help="Only tokenize the dataset and exit. Expected to be used"
                             "in combination with --cached_tokenization.",
                        action='store_true')
    return parser

def tokenizer_for_model(model_id: str = "google/gemma-2-2b"):
    tokenizer = AutoTokenizer.from_pretrained(config_utils.get_model_path(model_id))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

def load_hrcak_dataset(file_path):
    return load_dataset("text", data_files=file_path, split="train")


class TokenizerWrapper():
    '''
    Utility class that encapsulates a huggingface tokenizer, and the tokenization options and operations.
    '''

    def __init__(self, model_id, max_seq_length: int = 512):
        tokenizer = AutoTokenizer.from_pretrained(config_utils.get_model_path(model_id))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length=max_seq_length
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples):
        encoding = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None
        )
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    def tokenize_dataset(self, dataset):
        print("Tokenizing dataset...")
        return dataset.map(
            self,
            batched=True,
            num_proc=8,
            load_from_cache_file=False,
            keep_in_memory=True,
            remove_columns=["text"]
        )

    def save_tokenized_dataset(self, tokenized_dataset, save_folder):
        tokenized_dataset = tokenized_dataset.with_format("torch", columns=["input_ids", "attention_mask"])
        tokenized_dataset.save_to_disk(save_folder)

    def load_tokenized_dataset(self, save_folder):
        return load_from_disk(save_folder)

    def tokenize_cache_dataset(self, dataset, cache_label: str = None):
        '''
        :param dataset: a huggingface dataset
        :param cache_label: an id for caching the tokenized dataset, if None, no caching is done,
            if a string, the dataset is loaded from this folder, or if the folder does not exist,
            the result is saved there for subsequent use.
        '''
        print(f"Attempting to tokenize dataset, cache label = {cache_label}")
        if not cache_label or not get_tokenizer_cache_folder(cache_label, create=False):
            result = self.tokenize_dataset(dataset)
            if cache_label:
                save_folder = get_tokenizer_cache_folder(cache_label, create=True)
                if save_folder:
                    print(f"Saving tokenized dataset to {save_folder}")
                    self.save_tokenized_dataset(result, save_folder)
        else:
            print(f"Attempting to load tokenized dataset from {cache_label}")
            cache_folder = get_tokenizer_cache_folder(cache_label)
            try:
                result = self.load_tokenized_dataset(cache_folder)
                print(f"Loaded tokenized dataset from {cache_folder}")
            except:
                result = self.tokenize_dataset(dataset)
        return result

def create_model(model_id: str, quantize_params: dict, peft_params: dict):
    if 'gemma-2' in model_id.lower():
        # gemma2 works better with non-optimized attention implementation
        attn_type_kwargs = {'attn_implementation': 'eager'}
    else:
        attn_type_kwargs = {}
    if quantize_params['enabled']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantize_params['load_in_4bit'],
            bnb_4bit_quant_type=quantize_params['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quantize_params['bnb_4bit_use_double_quant'],
        )
        model = AutoModelForCausalLM.from_pretrained(
            config_utils.get_model_path(model_id),
            quantization_config=bnb_config,
            device_map=None,
            trust_remote_code=True,
            **attn_type_kwargs
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config_utils.get_model_path(model_id),
            device_map=None,
            trust_remote_code=True,
            **attn_type_kwargs
        )
        model.gradient_checkpointing_enable()
    if peft_params['enabled']:
        lora_config = LoraConfig(
            r=peft_params['r'],
            lora_alpha=peft_params['lora_alpha'],
            target_modules=peft_params['target_modules'],
            lora_dropout=peft_params['lora_dropout'],
            bias=peft_params['bias'],
            task_type=peft_params['task_type']
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model

def dataset_loader(label_or_path:str) -> DatasetDict:
    '''
    Load a dataset for training.
    :param label_or_path: either a known label (with a factory method), or a valid path to a hf dataset
    :return: hf dataset containing train and validation splits
    '''
    if label_or_path == 'macocu_v1': return get_macocu_text_v1()
    elif label_or_path == 'test_cro': return get_test_cro_dataset()
    elif os.path.exists(label_or_path):
        return load_from_disk(label_or_path)
    else:
        raise ValueError(f"Dataset argument must be either know label or existing hf dataset path: {label_or_path}")

def compute_perplexity_metric(eval_pred):
    print("Computing perplexity...")
    logits, labels = eval_pred
    # Shift logits and labels for causal language modeling
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    # Calculate cross-entropy loss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(shift_logits, shift_labels)
    # Compute perplexity from loss
    perplexity = np.exp(loss.detach().cpu().numpy())
    print(f"Perplexity: {perplexity}")
    return {"perplexity": perplexity}

def set_default_device(params: Dict):
    ''' Set default torch device from params if defined, else use cuda if available, or cpu. '''
    if 'device' in params: device = params['device']
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError("CUDA device not available.")
    torch.set_default_device('cuda')

def generate_cache_label(params: Dict, split: str) -> str:
    '''
    Label and folder name for caching tokenized datasets.
    It should be unique for relevant elements of given parameter contex -
    at least for the model and the dataset.
    '''
    # assume these ara paths (or hierarchical hf model ID), and take the last part as the name
    model_name, dset_name = Path(params['model_id']).name, Path(params['dataset_label']).name
    return f"tokenized_{model_name}_{dset_name}_{split}"

def setup_and_run_training(params: Dict):
    # 'device' param can break deepspeed run, so best to add it if needed for a particular machine
    if 'device' in params: set_default_device(params)
    model_id = params['model_id']
    # todo add hf login as a parameter; until then, uncomment first time when using, to download the model
    #if 'gemma' in model_id.lower(): config_utils.huggingface_login()
    if 'max_seq_length' not in params.keys(): params['max_seq_length'] = 512 # Set default, TODO: Remove
    tokenizer_wrapper = TokenizerWrapper(model_id, max_seq_length=params['max_seq_length'])
    dataset = dataset_loader(params['dataset_label'])
    if isinstance(dataset, DatasetDict) and 'train' in dataset and 'validation' in dataset:
        train_dataset = discard_columns(dataset['train'])
        val_dataset = discard_columns(dataset['validation'])
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    else: # single dataset
        train_dataset = discard_columns(dataset)
        val_dataset = None
        print(f"Training dataset size: {len(train_dataset)}")
    # tokenization
    if params['cached_tokenization'] is True:
        cache_label = generate_cache_label(params, 'train')
        cache_label_valid = generate_cache_label(params, 'valid')
    else: cache_label, cache_label_valid = None, None
    tokenized_train = tokenizer_wrapper.tokenize_cache_dataset(train_dataset, cache_label)
    tokenized_val = tokenizer_wrapper.tokenize_cache_dataset(val_dataset, cache_label_valid) if val_dataset else None
    # stop here if only tokenization is needed, ex. before the main training run
    if params['tokenize_only'] is True: return
    model = create_model(model_id, params['quantize'], params['peft'])
    if 'MODEL_TRAINING_OUTPUT' in params:
        config_utils.set_models_output_dir(params['MODEL_TRAINING_OUTPUT'])
    if 'deepspeed' in params:
        deepspeed_config = config_utils.get_deepspeed_config(params['deepspeed'])
        print(f"Using deepspeed config:\n{deepspeed_config}")
    else: deepspeed_config = None
    do_training(model,  tokenizer_wrapper.tokenizer, tokenized_train, tokenized_val,
                params, deepspeed_config)

def do_training(model, tokenizer, tokenized_train, tokenized_val,
                params, deepspeed_config=None):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_dir = config_utils.get_models_output_dir() / params['output_dir_tag']
    logging_dir = config_utils.get_models_output_dir() / params['logging_dir_tag']
    # TODO - if deepspeed is used, its parameters (like batch size) should be synced with the training arguments
    # temporary solution: corresponding parameters should be the same in both configurations
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to='tensorboard',
        per_device_train_batch_size=params['per_device_train_batch_size'],
        gradient_accumulation_steps=params['gradient_accumulation_steps'],
        per_device_eval_batch_size=params['per_device_eval_batch_size'],
        num_train_epochs=params['num_train_epochs'],
        max_steps=params['max_steps'],
        logging_strategy="steps", logging_steps=params['logging_steps'],
        eval_strategy=params['eval_strategy'], eval_steps=params['eval_steps'],
        eval_accumulation_steps=4,
        prediction_loss_only=False,
        save_strategy="steps", save_steps=params['save_steps'],
        save_total_limit=params['save_total_limit'],
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        max_grad_norm=params['max_grad_norm'],
        warmup_ratio=params['warmup_ratio'],
        group_by_length=True,
        lr_scheduler_type="cosine" if 'lr_scheduler_type' not in params else params['lr_scheduler_type'],
        fp16=params['fp16'],
        bf16=params['bf16'],
        remove_unused_columns=params['remove_unused_columns'],
        gradient_checkpointing=params['gradient_checkpointing'],
        deepspeed=deepspeed_config,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_perplexity_metric,
    )
    # custom barrier callback: when training ends, every process waits.
    # switched off for now, does not help -> sync error is not checkpoint saving anymore, but log saving
    # trainer.add_callback(FinalSyncCallback())
    # start training
    resume = True
    while True:
        try:
            trainer.train(resume_from_checkpoint=resume)
            print("Training finished.")
            if dist.is_initialized():
                time.sleep(30)
                dist.barrier()
            break
        except (ValueError, FileNotFoundError) as e:
            if resume: # try to train without resuming from checkpoint
                # if the error is due to something else, it should occur again
                print("No valid checkpoint found, training from scratch.")
                resume = False
                continue
            else:
                raise e
    def cleanup_and_rename():
        ''' delete checkpoints and rename the output and logging directories '''
        if dist.is_initialized() and dist.get_rank() != 0:
            return  # Non-main processes skip cleanup
        timetag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        remove_hf_checkpoints(output_dir)  # remove all checkpoint folders
        output_dir.rename(config_utils.get_models_output_dir() / f"model_{params['model_label']}_{timetag}")
        logging_dir.rename(config_utils.get_models_output_dir() / f"logs_{params['model_label']}_{timetag}")
    if dist.is_initialized():
        dist.barrier()  # sync all processes before saving
        if dist.get_rank() == 0: # save only once, for the main process
            trainer.save_model()
            cleanup_and_rename()
        dist.barrier()
        destroy_process_group() # clean up distributed training resources
    else:
        trainer.save_model()
        cleanup_and_rename()

def run_training_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    setup_and_run_training(params)

def main(**kwargs):
    """Run training experiment."""
    kwargs = config_utils.get_experiment_arguments(**kwargs)
    setup_and_run_training(kwargs)

if __name__ == "__main__":
    if len(sys.argv) < 2: # experimental code
        # print(get_test_cro_dataset())
        # get_bertic_dataset()
        pass
    else:
        yaml_file = sys.argv[1]
        run_training_yaml(yaml_file)