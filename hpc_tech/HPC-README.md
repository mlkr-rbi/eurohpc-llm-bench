These are the instructions based on our experience with MareNostrum 5 Accelerate and Leonardo Booster supercomputers.
On both machines the basic workflow was: setup the environment, setup the SLURM script, run the jobs. 
Environment setup relies on the provided modules (the linux module system) that contain basic 
libraries and tools such as python, cuda, mpi, compilers etc. Upon loading the models, the next step 
is to install the python packages, either in the  local py environment, 
a venv environment, conda environment or another type of environment enabled on the machine.

Before running on HPC, access to the cluster must be established, according to the provider's instructions.
Since the access config for Leonardo is a bit elaborate, we add succinct instructions in leonardo.connecting.txt.

The recipes below worked in March 2025 for both clusters, but they might become outdated if 
the clusters are reconfigured. In that case we expect that the procedure described here
will be similar and easily updatable.

The code2cluster.sh script is a helper script for copying the project code to the cluster via ssh, 
optionally not overwriting the setup files on the cluster. See the script for usage.

# ENVIRONMENT SETUP
- MN 5 ACC
- - the module environment is in mn-load-modules-py311-5.sh
- - since the cluster does not allow access to the outside internet, we downloaded
the pip packages locally using the following procedure:
- - - created a conda environment that matches the above module environment
- - - downloaded the pip packages, by running (in the conda environment):
"pip download -d py311-deepspeed -r requirements.txt"
 - - then, copy the py311 folder to the cluster and run:
"pip install --user -v --ignore-installed --no-index --no-build-isolation --find-links=./py311-deepspeed -r requirements.txt"
- - note that we installed in the local native python environment, not in a venv on conda
so we don't know whether these would've worked

- Leonardo Booster
- - the module environment is in leo_setup_module_env.sh
- - we attempted a venv setup (as recommended) but couldn't get it to work with deepspeed, 
as the deepspeed command ran the python interpreter from the system python, not the venv python
- - therefore we used the conda environment to install the package:
- - - created a conda env with python 3.11
- - - loaded the modules from the leo_setup_module_env.sh script
- - - ran pip install -r requirements.txt

# RUNNING THE JOBS
- MN 5 ACC
- - template for the slurm script is in /run.slurm.mn.template
- - adapt it with project/user id, module script, resource params, and other params 
- Leonardo Booster
- - template for the slurm script is in /run.slurm.leo.template
- - adapt it with project/user id, module script, resource params, conda env, and other params

- run the slurm batch job:
- - sbatch run.slurm // run.slurm is the adapted template
- - squeue gives job status: squeue --user=$(whoami)
- - slurm/*.txt files contain std. error and output redirects (the location can be
configured in run.slurm)