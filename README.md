# gemma2-challenge
Gemma2-challenge for kaggle competition

# environment setup, for a machine with internet access (not a HPC cluster)
- run "generate_requirements.sh" to generate requirements.txt automatically, by scanning the code
- optionally, first edit generate_requirements.sh to add more packages 'by hand'
- run "pip install -r requirements.txt" to install all the required packages

- copy .gitignore.template to .gitignore and modify it as needed
- copy settings.template.py to settings.py and enter the values of the variables that will be used

- create new directories:
    - datasets - to hold datasets
    - models - to hold models
    - outputs - to hold outputs (for evaluation scores)

# code overview
- src/ contains the original training code by Filip and Miha
- data_tools contains tools and utils for creating and manipulating the datasets
- utils package is for various utility modules not located in more specific packages 
- experiments - contains yaml configurations for various experiments - create your own by copying existing ones and changing values
- prompts - contains yaml configurations for building prompts

## training
- training package contains the training code
- settings.MODEL_TRAINING_OUTPUT is the directory where the training output will be saved
- full_training_v1.py is a refactored and slightly updated version of DPmultiGPU.py
- full_training_v2.py is a version of full_training_v1.py that loads params from a .yaml file
- run_module.sh is a util script for server run that runs a module in the background and redirects the output

## evaluation
- metrics - contains functionality for evaluation metrics
- evaluation - contain the main functionality for evaluation of HF models

## running full_training_v2.py with run_module.sh and using a .yaml file (from the root directory)
- `./run_module.sh training/full_training_v2.py training/setup/train_setup_v1.yaml`

## running any action through main.py
### training
- `python main.py --experiment experiments/mt-train-2b-peft-3epochs_lr2e-4.yml`
- `CUDA_VISIBLE_DEVICES=0,1 python main.py --experiment experiments/mt-train-2b-peft-3epochs_lr2e-4.yml`
### evaluation
- `python main.py --experiment mt-eval-2b-it-v000.yml`
- `CUDA_VISIBLE_DEVICES=0,1 python main.py --experiment experiments/mt-eval-2b-peft-3epochs_lr2e-4-2024-12-01-v000.yml --max_examples 10`

# HPC setup
- hpc_tech folder contains scripts and instructions for setting up and running the code on the HPC cluster
- scripts here are not refactored and generalized yet, but are ad hoc tools
- for now only some util scripts, more elaborate instructions to come
- more info can be found on MareNostrum machine, in the project root folder: 
- - /gpfs/projects/ehpc124/ , especially in the README.md file and the scripts folder
- - you can create ~/bin folder and put your custom util scripts there, it is added to PATH automatically

# DEEPSPEED SETUP
- full_training_v3.py is a version of full_training_v2.py that uses deepspeed
- to use deepspeed, a 'deepspeed' parameter needs to be present, defined either on cmdline 
- - the value of the parameter is either a path (same convention as for .yml settings), or a json string 
- - for example: `python main.py --experiment mt-train-local-test.yml --deepspeed deepspeed_test.json`
- - YAML example is in experiments/mt-train-local-test-deepspeed.yml
- Installing deepspeed locally: it depends on MPI and mpy4py
  - the following works on ubuntu 24.04
  - if MPI is not installed, install it with `sudo apt install mpich libmpich-dev`
  - install mpy4py with
    - export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu (or similar, because otherwise compilation will "fail to link") 
    - `pip install mpi4py`
  - do `pip install deepspeed`
- Installing deepspeed on marenostrum
- - from scratch: same as with the base environment, once folder with packages is created
- - into the existing environment, based on a folder with new packages added locally:
first activate the environment: module loads' and maybe venv ...
"pip install --user -v --ignore-installed --no-index --no-build-isolation --find-links=./py311-deepspeed mpi4py deepspeed"

# DEEPSPEED RUN
- create the project folder with the code, either by copying the 'official' repo 
via script, or using hpc_tech/mn_push_repo.sh script to create a custom location
- setup the environment: edit settings.py, and if necesary, the .yml file in the experiments/
folder, if for the wiki dataset the 'dataset_label' property needs to point to the 
full path of the dataset (no loader method yet)
- copy run.slurm.MACHINE_ID.template to run.slurm and customize, if needed:
- - the resources requested (nodes, gpu, time)
- - the code that sets up the environment (module loads, ...) 
- - the experiment file that will be used
- run the slurm batch job:
- - sbatch run.slurm
- - squeue gives job status: squeue --user=$(whoami)
- - slurm/*.txt files contain std. error and output redirects (the location can be
configured in run.slurm)