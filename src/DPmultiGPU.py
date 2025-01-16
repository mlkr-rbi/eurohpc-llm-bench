import os
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
from datasets import load_dataset
from accelerate import Accelerator

# Load Hugging Face token and login
with open("/workspace/gemma2/tok_gemma2.txt", "r") as f:
    huggingface = f.read().strip()
login(token=huggingface)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of CUDA Devices: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Model and tokenizer setup
model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load dataset
def load_text_dataset(file_path):
    return load_dataset("text", data_files=file_path, split="train")

file_path = '/workspace/gemma2/data/pretraining/hrca-transformer.txt'
dataset = load_text_dataset(file_path)

# Tokenization
max_seq_length = 512

def tokenize_function(examples):
    encoding = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors=None
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

# Process dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"]
)

# Split dataset
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Model setup with proper gradient handling
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for k-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="/workspace/gemma2/output/LoRA-all-bs4-2gpu",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    prediction_loss_only=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()
