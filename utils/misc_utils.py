import os
import shutil


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