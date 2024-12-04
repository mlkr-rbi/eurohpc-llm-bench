''' training in torch, basically a more structured version of DPmultiGPU.py code '''

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from pyarrow import set_timezone_db_path

from data_tools.dataset_factory import get_bertic_dataset, get_macocu_v1, get_test_cro_dataset
import settings
from settings import MODEL_TRAINING_OUTPUT, HUGGINGFACE_TOKEN

import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset, DatasetDict


# Load Hugging Face token and login
def huggingface_login():
    login(token=HUGGINGFACE_TOKEN)

def tokenizer_for_model(model_id: str = "google/gemma-2-2b"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

def load_hrcak_dataset(file_path):
    return load_dataset("text", data_files=file_path, split="train")


class TokenizerWrapper():
    '''
    Utility class that encapsulates a huggingface tokenizer, and the tokenization options and operations.
    '''

    def __init__(self, model_id, max_seq_length: int = 512):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
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
        return dataset.map(
            self,
            batched=True,
            num_proc=4,
            remove_columns=["text"]
        )

def create_model(model_id: str, quantize_params: dict, peft_params: dict):
    if quantize_params['enabled']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantize_params['load_in_4bit'],
            bnb_4bit_quant_type=quantize_params['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quantize_params['bnb_4bit_use_double_quant'],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True
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

def dataset_loader(label:str) -> DatasetDict:
    '''
    Load a dataset for training.
    :return: hf dataset containing train and validation splits
    '''
    if label == 'macocu_v1': return get_macocu_v1()
    elif label == 'test_cro': return get_test_cro_dataset()
    else: raise ValueError(f"Unknown dataset label: {label}")

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

def setup_and_run_training(params):
    tokenizer_wrapper = TokenizerWrapper(params['model_id'])
    dataset = dataset_loader(params['dataset_label'])
    if isinstance(dataset, DatasetDict) and 'train' in dataset and 'validation' in dataset:
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    else: # single dataset
        train_dataset = dataset
        val_dataset = None
        print(f"Training dataset size: {len(train_dataset)}")
    tokenized_train = tokenizer_wrapper.tokenize_dataset(train_dataset)
    tokenized_val = tokenizer_wrapper.tokenize_dataset(val_dataset) if val_dataset else None
    model_id = params['model_id']
    if 'gemma' in model_id.lower(): huggingface_login()
    model = create_model(model_id, params['quantize'], params['peft'])
    if 'MODEL_TRAINING_OUTPUT' in params:
        settings.MODEL_TRAINING_OUTPUT = params['MODEL_TRAINING_OUTPUT']
    do_training(model, tokenizer_wrapper.tokenizer, tokenized_train, tokenized_val, params)

def do_training(model, tokenizer, train_dataset, val_dataset, params):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    output_dir = Path(MODEL_TRAINING_OUTPUT) / params['output_dir_tag']
    logging_dir = Path(MODEL_TRAINING_OUTPUT) / params['logging_dir_tag']
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to='tensorboard',
        per_device_train_batch_size=params['per_device_train_batch_size'],
        gradient_accumulation_steps=params['gradient_accumulation_steps'],
        per_device_eval_batch_size=params['per_device_eval_batch_size'],
        num_train_epochs=params['num_train_epochs'],
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
        lr_scheduler_type="cosine",
        fp16=params['fp16'],
        bf16=params['bf16'],
        remove_unused_columns=params['remove_unused_columns'],
        ddp_find_unused_parameters=params['ddp_find_unused_parameters'],
        gradient_checkpointing=params['gradient_checkpointing'],
        local_rank=-1
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_perplexity_metric,
    )
    resume = True
    while True:
        try:
            trainer.train(resume_from_checkpoint=resume)
            print("Training finished.")
            break
        except ValueError as e:
            msg = str(e)
            if "no valid checkpoint found" in msg.lower():
                print("No valid checkpoint found, training from scratch.")
                resume = False
                continue
            else:
                raise e
    timetag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.rename(Path(MODEL_TRAINING_OUTPUT) / f"model_{params['model_label']}_{timetag}")
    logging_dir.rename(Path(MODEL_TRAINING_OUTPUT) / f"logs_{params['model_label']}_{timetag}")

def run_training_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        params = yaml.safe_load(file)
    setup_and_run_training(params)

if __name__ == "__main__":
    if len(sys.argv) < 2: # experimental code
        # print(get_test_cro_dataset())
        # get_bertic_dataset()
        pass
    else:
        yaml_file = sys.argv[1]
        run_training_yaml(yaml_file)