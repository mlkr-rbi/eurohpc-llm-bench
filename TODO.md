# Main tech-related TODO LIST

# TODO: debugging the code until a model is trained in a deepspeed 'regime'
- a priority over next step
- install deepspeed and mpi4py first in your env: see readme or below
- solve the 'device_map' task, or just edit local files on the server, or fix to None in github as default
- this is an iterative process, the training is run using 
sbatch run.slurm (see run.slurm.template)
- output of the process is in slurm error and output files (see template) 
- it seems that the only thing left is to configure deepspeed and hf/torch params properly

# TODO: scripts for precise monitoring of resource usage (memory, cpu, gpu) accross nodes, for a specifed job
- there should be slurm facilities to do this

# improved running the code using deepspeed
- current setup uses within-python deepspeed, ie 
Trainer is configured to use deepspeed and the program is run via
python main.py
- - todo: see if using the 'deepspeed' command (or the 'accelerate') command
from command line is better is some way, it could be according to 
some web materials

# code mods for distributed training
- TODO: introduce parameter for 'device_map'
- - upon model loading, device_map='auto' is good for inference, while for
distributed training, device_map=None should be used (or maybe something else)

# environment setup
- current deepspeed environment was created ad-hoc
- - first, py311 was copied locally, then:
"pip download --exists-action i -d py311/ --find-links py311/ mpi4py deepspeed nvidia-ml-py"
- - this also downloads NEW versions of existing packages, but this does not seem to be a problem for install
- - folder with new packages was copied to marenostrum, then (witin a 3.11 environment): 
"pip install --user -v --ignore-installed --no-index --no-build-isolation --find-links=./py311-deepspeed mpi4py deepspeed"
- - the 3.11 environment will be in the environments folder, once the write access is granted

- todo: replicate this, ie, create a local environment that works for deepspeed
- todo: try creating a complete folder with packages from scratch