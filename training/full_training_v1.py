''' training in torch, basically a more structured version of DPmultiGPU.py code '''

import os
from datetime import datetime
from pathlib import Path

import numpy as np

from settings import MODEL_TRAINING_OUTPUT

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import torch.nn as nn
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
from datasets import load_dataset, Dataset
from accelerate import Accelerator

# Load Hugging Face token and login
def huggingface_login():
    with open("/workspace/gemma2/tok_gemma2.txt", "r") as f:
        huggingface = f.read().strip()
    login(token=huggingface)

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

def create_model(model_id: str = "google/gemma-2-2b", quantize=True, peft=True):
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    else:
        bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    if quantize:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    if peft:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model

def get_bertic_dataset(subsample=5000, rseed=5439):
    dset = load_dataset('classla/xlm-r-bertic-data', trust_remote_code=True)
    dset = dset['train'] # sample efficiently!
    if subsample and subsample < len(dset):
        dset = dset.shuffle(seed=rseed)
        dset = dset.select(range(subsample))
    dset = dset['output']
    # clean whitespaces, make new hf dataset
    dset = Dataset.from_list([{ 'text': txt.strip() } for txt in dset])
    return dset

def get_test_cro_dataset(subsample=5000, rseed=5439):
    dset = load_dataset('saillab/alpaca-croatian-cleaned')
    dset = dset['train']
    if subsample and subsample < len(dset):
        dset = dset.shuffle(seed=rseed)
        dset = dset.select(range(subsample))
    dset = dset['output']
    # clean whitespaces, make new hf dataset
    dset = Dataset.from_list([{ 'text': txt.strip() } for txt in dset])
    return dset

def setup_and_run_training(model_id, model_label, dataset: Dataset = None, production=False):
    tokenizer_wrapper = TokenizerWrapper(model_id)
    tokenized_dataset = tokenizer_wrapper.tokenize_dataset(dataset)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    # model
    model = create_model(model_id)
    do_training(model, tokenizer_wrapper.tokenizer, train_dataset, val_dataset,
                model_label=model_label, production=production)

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

def do_training(model, tokenizer, train_dataset, val_dataset, model_label, production=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    # if the folders are present, the training will be restarted
    output_dir_tag = f"model_{model_label}"
    output_dir = Path(MODEL_TRAINING_OUTPUT)/output_dir_tag
    logging_dir_tag = f"logs_{model_label}"
    logging_dir = Path(MODEL_TRAINING_OUTPUT)/logging_dir_tag
    if production: # set params for production-level long runs
        prediction_loss_only = False
        compute_metrics = compute_perplexity_metric
        resume_from_checkpoint = True
        num_train_epochs = 3
        gradient_accumulation_steps = 8
    else: # set params for short test runs
        prediction_loss_only = True
        compute_metrics = None
        resume_from_checkpoint = False
        num_train_epochs = 0.05
        gradient_accumulation_steps = 1
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to='tensorboard',
        per_device_train_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        logging_strategy="steps", logging_steps=10,
        # evaluation on eval dataset
        eval_strategy="steps", eval_steps=100,
        eval_accumulation_steps=4,
        prediction_loss_only=prediction_loss_only,
        # checkpointing
        save_strategy="steps", save_steps=500,
        save_total_limit=2,
        # optimizer
        learning_rate=2e-4,
        weight_decay=0.001,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        # misc
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        local_rank=-1
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    while True:
        try:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            print("Training finished.")
            break
        except ValueError as e:
            msg = str(e)
            if "no valid checkpoint found" in msg.lower():
                print("No valid checkpoint found, training from scratch.")
                resume_from_checkpoint = False
                continue
            else:
                raise e
    # save the model and the log to timestamped folders (rename the folders)
    timetag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.rename(Path(MODEL_TRAINING_OUTPUT)/f"model_{model_label}_{timetag}")
    logging_dir.rename(Path(MODEL_TRAINING_OUTPUT)/f"logs_{model_label}_{timetag}")

if __name__ == "__main__":
    #setup_and_run_training(model_id='HuggingFaceTB/SmolLM-135M', model_label='SmolLM-135M', dataset=get_test_cro_dataset())
    #print(get_test_cro_dataset())
    get_bertic_dataset()