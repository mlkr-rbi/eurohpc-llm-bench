from typing import Union, Any, Dict, List, Optional
import os
from pathlib import Path
import yaml
import json
import datetime
import time
import pandas as pd
from huggingface_hub import login
import warnings

import settings


def raise_warning(message: str="Warning", stacklevel=2):
    warnings.warn(message, stacklevel=stacklevel)


def raise_deprecated_warning_decorator(function):
    def f(*args, **kwargs):
        message = f"The method '{function.__name__}' is deprecated and will be removed!"
        raise_warning(message, stacklevel=3)
        return function(*args, **kwargs)
    return f


def huggingface_login():
    """Load Hugging Face token and login"""
    try:
        login(token=settings.HUGGINGFACE_TOKEN)
    except:
        raise_warning("HUGGINGFACE login FAILED!")


def get_cwd() -> Path:
    return Path(os.getcwd())


def get_app_root_path() -> Path:
    """Get the application root path. A parent directory of settings.py file."""
    return Path(settings.__file__).parent


def handle_abs_relative_path(path: Union[str, Path],
                             parent_paths: List[Union[str, Path]]) -> Path:
    """Handles path given with relative or absolute positioning.

    Args:
        path (Union[str, Path]): Relative or absolute path.
        parent_paths (List[Union[str, Path]]): Absolute path of possible parent directories.

    Raises:
        ValueError: If arg 'parent_path' is not absolute path.

    Returns:
        Path: Absolute path of the file.
    """
    for parent_path in parent_paths:
        if not os.path.isabs(str(parent_path)):
            msg = "Arg 'parent_path' must be absolute path! Given path: ", str(parent_path)
            raise ValueError(msg)
    if os.path.isabs(str(path)):
        return Path(path)
    else:
        # Check if any path exists - and use the first that exists.
        for parent_path in parent_paths:
            p = Path(str(parent_path)) / path
            if p.exists(): return Path(str(parent_path)) / path
    # Use the last parent path as parent directory if no combination exists.
    return p

def get_absolute_path(path: str) -> Path:
    """Get absolute path from given relative path string."""
    return handle_abs_relative_path(path, [get_app_root_path()])


def get_path_with_suffix(path: Union[str, Path], suffix: str=".yml") -> Union[str, Path]:
    """Get path with added suffix."""
    if isinstance(path, str) and not path.endswith(".yml"):
        path = path + suffix
    if isinstance(path, Path):
        path = path.with_suffix(suffix)
    return path


def get_config(path: Union[Path, str]) -> Any:
    """Get configuration from YAML file."""
    return yaml.safe_load(open(path))


def save_config(config_data: Any, file_path: Union[Path, str]):
    """Save configuration to YAML file."""
    with open(file_path, "w") as file:
        yaml.safe_dump(config_data, file)


def save_scores(scores: Any, path: Union[Path, str]):
    with open(path, 'w') as file:
        file.write(json.dumps(scores, sort_keys=True, indent=4))


# def get_settings(path="settings.yml") -> Dict:
#     """Get default settings configuration."""
#     return get_config(get_absolute_path(path))


def get_models_output_dir() -> Path:
    """Get default directory to store new trained models."""
    return get_absolute_path(settings.MODEL_TRAINING_OUTPUT)


def set_models_output_dir(models_output_path: str, Path):
    """Set default directory to store new trained models."""
    settings.MODEL_TRAINING_OUTPUT = str(models_output_path)


def get_models_dir() -> Path:
    """Get default directory to store custom models."""
    return get_absolute_path(settings.MODELS_DIR)


def get_model_path(model_path: str) -> Path:
    """Get default directory to store custom models."""
    parent_paths = [get_cwd(), get_app_root_path(), get_models_dir()]
    return handle_abs_relative_path(model_path, parent_paths)


def get_datasets_dir() -> Path:
    """Get default directory to store custom datasets."""
    return get_absolute_path(settings.DATASETS_DIR)


def get_outputs_dir() -> Path:
    """Get default directory to store outputs."""
    return get_absolute_path(settings.OUTPUTS_DIR)


__experiment_output_dir: Path = None
def create_experiment_output_dir(experiment: str=None):
    """Create default directory to store outputs for current experiment.

    Args:
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None.
    """
    global __experiment_output_dir
    experiment = Path(experiment).stem
    idx = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    exp_file_name = experiment + "-" + idx
    path = get_outputs_dir() / exp_file_name
    if path.exists():
        time.sleep(1)
        return get_experiment_output_dir(experiment=experiment)
    path.mkdir(parents=True)
    __experiment_output_dir = path


def get_experiment_output_dir(experiment: str=None, reset: bool=False) -> Path:
    """Get path to the experiment output directory.

    Args:
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None.
        reset (bool, optional): Set True to create new experiment output file. Defaults to False.
    Returns:
        Path: Path to the experiment output directory.
    """
    global __experiment_output_dir
    if __experiment_output_dir == None:
        create_experiment_output_dir(experiment)
    return __experiment_output_dir


def get_experiment_output_prediction_file(dataset_name: str,
                                          experiment_output_dir: Path=None,
                                          experiment: str=None) -> Path:
    """Get path to the prediction file for given dataset in experiment output directory.

    Args:
        dataset_name (str): The name of dataset for which we want prediction file.
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None.

    Returns:
        Path: Path to the prediction file for given dataset.
    """
    if experiment_output_dir == None:
        experiment_output_dir = get_experiment_output_dir(experiment=experiment)
    return experiment_output_dir / (dataset_name + ".csv")


def get_experiment_output_prediction(dataset_name: str,
                                     experiment_output_dir: Path=None,
                                     experiment: str=None) -> Dict[str, List[str]]:
    """Get inputs, outputs and predictions for given dataset in experiment output directory.

    Args:
        dataset_name (str): The name of dataset for which we want prediction file.
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None.

    Returns:
        Dict[str, List[str]]: Dictionary with inputs, outputs and predictions for given dataset. 
    """
    pred_file = get_experiment_output_prediction_file(dataset_name, experiment_output_dir, experiment)
    df = pd.read_csv(pred_file)
    ans = {
        'inputs':      df['inputs'].to_list(),
        'outputs':     df['outputs'].to_list(),
        'predictions': df['predictions'].to_list(),
    }
    return ans


def save_experiment_output_prediction(inputs: List[str],
                                      outputs: List[str],
                                      predictions: List[str],                                     
                                      dataset_name: str,
                                      experiment_output_dir: Path=None,
                                      experiment: str=None):
    """Save inputs, outputs and predictions for given dataset in experiment output directory.

    Args:
        inputs (List[str]): List of input texts.
        outputs (List[str]): List of output texts. References or golden standard.
        predictions (List[str]): List of predictions, ie. texts generated by the model.
        dataset_name (str): The name of dataset for which we want prediction file.
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None.
    """
    pred_file = get_experiment_output_prediction_file(dataset_name, experiment_output_dir, experiment)
    l = min(len(inputs), len(outputs), len(predictions))
    d = {
        'inputs':      inputs[:l],
        'outputs':     outputs[:l],
        'predictions': predictions[:l],
    }
    df = pd.DataFrame(d)
    df.to_csv(pred_file)


def get_experiment_output_scores_file(dataset_name: str,
                                      experiment_output_dir: Path=None,
                                      experiment: str=None) -> Path:
    """Get path to the file with evaluation scores for given dataset in experiment output directory.

    Args:
        dataset_name (str): The name of dataset for which we want prediction file.
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None. Defaults to None.

    Returns:
        Path: Path to the file with evaluation scores for given dataset.
    """
    if experiment_output_dir == None:
        experiment_output_dir = get_experiment_output_dir(experiment=experiment)
    return experiment_output_dir / (dataset_name + "-scores.json")


def save_experiment_output_scores(scores: Any, dataset_name: str,
                                  experiment_output_dir: Path=None,
                                  experiment: str=None):
    """Save evaluation scores for given dataset in experiment output directory.

    Args:
        dataset_name (str): The name of dataset for which we want prediction file.
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None. Defaults to None.

    """
    path = get_experiment_output_scores_file(dataset_name, experiment_output_dir, experiment)
    save_scores(scores, path)


def get_experiment_output_config_file(experiment_output_dir: Path=None,
                                      experiment: str=None)-> Path:
    """Get path to the experiment configuration file in experiment output directory.

    Args:
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None. Defaults to None.

    Returns:
        Path: Path to the experiment configuration file.
    """
    if experiment_output_dir == None:
        experiment_output_dir = get_experiment_output_dir(experiment=experiment)
    return experiment_output_dir / "exp-config.yml"


def get_experiment_output_config(experiment_output_dir: Path=None) :
    """Get path to the experiment configuration file in experiment output directory.

    Args:
        experiment_output_dir (Path, optional): Experiment output directory path. Defaults to None.
        
    Returns:
        Path: Path to the experiment configuration file.
    """
    path = get_experiment_output_config_file(experiment_output_dir=experiment_output_dir)
    return get_config(path=path)


def save_experiment_output_config(config_data: Any,
                                  file_path: Union[Path, str]=None,
                                  experiment: str=None):
    """Save experiment configuration to YAML file in experiment output directory.

    Args:
        config_data (Any): Configuration data to save. Usually dictionary.
        file_path (Union[Path, str], optional): Path to the output configuration path. Defaults to None.
        experiment (str, optional): The name of the experiment created from experiment configuration file name. Defaults to None. Defaults to None.
    """
    print(experiment)
    if file_path == None:
        if experiment != None:
            file_path = get_experiment_output_config_file(experiment=experiment)
        else:
            file_path = get_experiment_output_config_file(experiment=config_data['experiment'])
    save_config(config_data, file_path)


def get_prompts_dir() -> Path:
    """Get default directory to prompt configuration files."""
    return get_absolute_path(settings.PROMPTS_DIR)


def get_prompts(prompt_name_or_path: str, instruction_lang: str='en') -> Dict:
    """Get prompt configuration."""
    path = get_path_with_suffix(prompt_name_or_path, ".yml")
    parent_paths = [get_cwd(), get_app_root_path(), get_prompts_dir()]
    config = get_config(handle_abs_relative_path(path, parent_paths))
    # Check if instruction language exist
    if instruction_lang not in config.keys():
        raise KeyError(
            f"There are no prompts for the given instruction language '{instruction_lang}'! " +
            f"Supported instruction languages are: {list(config.keys())}")
    return config[instruction_lang]


def get_experiments_dir() -> Path:
    """Get default directory to experiment configuration files."""
    return get_absolute_path(settings.EXPERIMENTS_DIR)


def get_experiment(experiment_name_or_path: str) -> Dict:
    """Get experiment configuration."""
    path = get_path_with_suffix(experiment_name_or_path, ".yml")
    parent_paths = [get_cwd(), get_app_root_path(), get_experiments_dir()]
    return get_config(handle_abs_relative_path(path, parent_paths))


def get_default_experiment(action: str=None) -> Dict:
    """Get default experiment configuration."""
    if action == 'evaluation':
        return get_experiment(settings.DEFAULT_EVALUATION_EXPERIMENT)
    elif action == 'training':
        return get_experiment(settings.DEFAULT_TRAINING_EXPERIMENT)
    return {}


def get_macocu_sentences_file_path():
    return get_absolute_path(settings.MACOCU_SENTENCES_FILE)


def get_macocu_pairs_dataset_dir():
    return get_datasets_dir() / settings.MACOCU_PAIRS_DATASET_NAME


def get_macocu_text_dataset_dir():
    return get_datasets_dir() / settings.MACOCU_TEXT_DATASET_NAME
    
    
@raise_deprecated_warning_decorator
def get_macocu_dataset_dir():
    return get_absolute_path(settings.MACOCU_DATASET_V1)


def get_experiment_arguments(action: str=None, **kwargs) -> Dict[str, Any]:
    # Get default experiment config
    exp_config = get_default_experiment(action)
    # Update experiment config from given experiment custom config file
    if 'experiment' in kwargs:
        custom_config = get_experiment(kwargs['experiment'])
        for k in custom_config:
            exp_config[k] = custom_config[k] # update config
    # Update experiment config with given arguments
    for k in kwargs:
        if kwargs[k] != None:
            exp_config[k] = kwargs[k] # update config
    # exp_config['experiment'] = kwargs['experiment']
    return exp_config


if __name__=="__main__":
    # Test utils functions
    print("get_experiments_dir:    ", get_experiments_dir())
    print("get_default_experiment: ", get_default_experiment())
    print("get_experiment:         ", get_experiment("mt-000"))
    print("get_prompts:            ", get_prompts("mt-hr-en", 'hr'))
    print("get_models_dir:         ", get_models_dir())
    print("get_model_path:         ", get_model_path("google/gemma-2-2b"))
    print("get_datasets_dir:       ", get_datasets_dir())
    print("get_outputs_dir:        ", get_outputs_dir())