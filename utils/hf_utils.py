import os
import shutil
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