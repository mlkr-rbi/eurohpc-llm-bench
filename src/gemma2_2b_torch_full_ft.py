import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*cache_implementation.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cpu.amp.autocast.*")
# Remove the TypeError warning filter since it's not a valid warning category
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load and process dataset
dataset = load_dataset("text", data_files='/home/fuljanic/gemma2/data/corpus-hrcak-cro-transformer.txt')['train']

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors=None
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
train_dataset, val_dataset = tokenized_dataset.train_test_split(test_size=0.1).values()

def train():
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configure model with bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    model.gradient_checkpointing_enable()
    
    # Data loading
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )
    
    # Optimizer with less conservative settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with shorter warmup
    num_epochs = 3
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 20
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Gradient accumulation steps
    gradient_accumulation_steps = 8
    max_grad_norm = 0.5
    
    # Training loop
    model.train()
    running_loss = []
    nan_counter = 0
    
    for epoch in range(num_epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass with bfloat16
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                nan_counter += 1
                print(f"\nWarning: NaN loss detected (occurrence {nan_counter}). Skipping batch.")
                if nan_counter >= 3:
                    print("\nToo many NaN losses. Stopping training.")
                    return
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation and optimization
            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update progress bar with smoothed loss
            running_loss.append(loss.item() * gradient_accumulation_steps)
            if len(running_loss) > 100:
                running_loss.pop(0)
            avg_loss = np.mean(running_loss)
            
            # Update progress bar
            progress.set_description(
                f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}"
            )
        
        # Save checkpoint after each epoch
        checkpoint_path = f"/home/fuljanic/gemma2/output_hrcak_9b/model_epoch_{epoch+1}"
        model.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train()