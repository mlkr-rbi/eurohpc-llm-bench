# EuroHPC LLM Benchmarking
The code in this repo is developed during the Benchmark Access Project granted
through the EuroHPC initiative for MareNostrum 5 ACC and Leonardo Booster supercomputers.
During the project we used the code to fine-tune LLMs on Croatian Wikipedia, and on the parallel Croatian-English translation corpora.
We release the code in the hope that it will be useful for other practitioners working on these machines, 
or any machine with a similar software stack (Linux, SLURM, CUDA, PyTorch, deepspeed, HuggingFace).

The goal of the project was to develop knowledge and code for running fine-tuning of LLMs on HPC clusters.
The project, initially inspired by the Gemma2-challenge on Kaggle, focused on fine-tuning LLMs for Croatian.
The repository contains LLM fine-tuning code (based on HuggingFace and deepspeed), configuration files, 
helper scripts, documentation of setup procedures, etc. 

The code worked on both MareNostrum ACC 5 and Leonardo Booster clusters in March 2025.
If we are granted access to these or similar clusters in the future, we will continue to develop the code.
Note that the code is work in progress and has been developed with a limited time budget.

The code is released under the Apache License 2.0 

# overview
- training - contains all the iterations of the training code
- evaluation - contains the machine translation evaluation code
- experiments - contains yaml configurations for various experiments and deepspeed setup files, 
these can be used as templates for new experiments
- data_tools contains dataset factory methods and utils for translation prompts
- utils package is for various utility modules not located in more specific packages 
- prompts - contains yaml configurations for building translation prompts
- src - contains the initial training code by Filip and Miha

# environment setup
- these instructions are to be used for setting up a local dev environment, 
or as generic guidelines for setting up the environment on HPC clusters.
Specific instructions for MareNostrum 5 and Leonardo are in the hpc_tech folder
- use requirements.txt to install the required packages
- alternatively, run "generate_requirements.sh" to generate requirements.txt automatically, by scanning the code
optionally, edit generate_requirements.sh to add more custom packages by hand
- run "pip install -r requirements.txt" to install all the required packages
- copy .gitignore.template to .gitignore and modify it as needed
- copy settings.template.py to settings.py and enter the values of the variables that will be used
make sure that all the relevant files and folders exist

# running the code
- the runs can be local runs, used for dev and debug, or runs on HPC clusters
- the main entry point for running the code is main.py
it receives an '--action' argument, which can be 'training' or 'evaluation', and
and '--experiment' argument, which is a name of a .yml file in the 'experiments' folder.
if using deepspeed, use '--deepspeed' argument to specify the deepspeed config file
there can be further command line arguments, depending on the action, 
and these arguments overwrite, ie, customize, the values in the .yml file
- for local runs, run the main.py script with appropriate arguments, or 
use deepspeed_main.sh, a script for running/testing deepspeed runs locally
- for hpc runs, use the run.slurm template scripts - more info in the 'hpc_tech' folder
- run_module.sh is a helper script for executing a module on a cluster, 
in the background, with outputs redirected to a file 

# training
- training package contains the training code
- settings.MODEL_TRAINING_OUTPUT is the directory where the training output will be saved
- full_training_v3.py is currently the main training script
- see the mt-train files in the experiments folder for examples of how to run the training

# evaluation
- evaluation package contains the evaluation code
- see mt-eval files in the experiments folder for examples of how to run the evaluation

# local deepspeed setup
Installing deepspeed locally, for debug - it depends on MPI and mpy4py
- the following works on ubuntu 24.04
- if MPI is not installed, install it with `sudo apt install mpich libmpich-dev`
- install mpy4py with
- - export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu (or similar, because otherwise compilation will "fail to link") 
- - `pip install mpi4py`
- do `pip install deepspeed`
- run the training or eval with deepspeed_main.sh script