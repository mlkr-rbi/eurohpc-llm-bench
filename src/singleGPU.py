
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
from huggingface_hub import login

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# Load Hugging Face CLI token
with open("/workspace/gemma2/tok_gemma2.txt", "r") as f:
    huggingface = f.read().strip()

login(token=huggingface)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "google/gemma-2-2b"
# Load the pretrained model and tokenizer, move the model to the correct device
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
file_path = '/workspace/gemma2/data/pretraining/hrca-transformer.txt'

# Step 1: Load the text file as a dataset
def load_text_dataset(file_path):
    """Loads a text dataset from a .txt file, with each line as a text entry."""
    return load_dataset("text", data_files=file_path, split="train")

dataset = load_text_dataset(file_path)

# Step 2: Tokenize the dataset and add labels
max_seq_length = 512

def tokenize_function(examples):
    """Tokenizes text examples and creates labels for causal language modeling."""
    encoding = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )
    encoding["labels"] = encoding["input_ids"].copy()  # Set labels as input_ids for causal LM
    return encoding

# Apply tokenization and label creation
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Step 3: Set up TrainingArguments and Trainer for pretraining
training_args = TrainingArguments(
    output_dir="./output/Lora-2-bs2",
    per_device_train_batch_size=2,  # Adjust based on GPU memory
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    prediction_loss_only=True,
    remove_unused_columns=False
)

# Split the tokenized dataset into training and validation datasets
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# %%
# Step 4: Initialize Trainer
# Set up PEFT LoRA for fine-tuning.
# target_modules=["q_proj", "o_proj", "k_proj", 
# "v_proj", "gate_proj", "up_proj", "down_proj"], 

lora_config = LoraConfig(
    r=4,
    target_modules=["k_proj", "v_proj", "q_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model = model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()