import os
import shutil, psutil
import torch.distributed as dist
from transformers import TrainerCallback

# Define a callback to force a barrier when training finishes.
class FinalSyncCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        if dist.is_initialized():
            # Wait until all processes have reached the training-end callback.
            dist.barrier()
        return control

def remove_hf_checkpoints(output_dir, checkpoint_prefix="checkpoint-"):
    """
    Remove all model checkpoint folders from the huggingface Trainer's output_dir
    :param output_dir: directory where checkpoints are stored
    :param checkpoint_prefix: folder is considered a checkpoint if it starts with this prefix
    """
    for folder_name in os.listdir(output_dir):
        if folder_name.startswith(checkpoint_prefix):
            checkpoint_path = os.path.join(output_dir, folder_name)
            shutil.rmtree(checkpoint_path)
            print(f"Deleted {checkpoint_path}")

def print_available_ram():
    total_memory = psutil.virtual_memory().total
    total_memory_gb = total_memory / (1024 ** 3)
    print(f"Total available RAM: {total_memory_gb:.2f} GB")
    free_memory = psutil.virtual_memory().free
    free_memory_gb = free_memory / (1024 ** 3)
    print(f"Free RAM: {free_memory_gb:.2f} GB")


def real_length_in_tokens(tok_output, tokenizer):
    '''
    For a hf tokenizer and its outputs, calculate the real length
    of the input prompt, in the number of tokens.
    :param tok_output: results of tokenizer(prompt)
    :param tokenizer: huggingface tokenizer
    '''
    input_ids = tok_output['input_ids'][0]
    eos_token_id = tokenizer.eos_token_id
    real_length = 0
    for token_id in input_ids:
        if token_id == eos_token_id:
            break
        real_length += 1
    return real_length
