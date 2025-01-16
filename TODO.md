# Main tech-related TODO LIST

# TODO: Evaluation

# TODO: debugging the code until a model is trained in a deepspeed 'regime'
- deploy new deepspeed training setup from the repo to mare nostrum, and test for correctness

# TODO: scripts for precise monitoring of resource usage (memory, cpu, gpu) accross nodes, for a specifed job
- check deepspeed logging functionality 
- check slurm logging functionality

# TODO: Logging
- check aggregation of logs (validation scores) - is it done per node or after gradient aggregation?
- create a solution (iteratively) for logging that encompasses: 
- - erorrs
- - custom log messages
- - deepspeed metrics and training params
- - resources, primarily GPU usage

# TODO: Environment
- Setup all on Leonardo

# TODO: Define final task

# TODO: Training and In-train Evaluation
- loss evaluation on eval_dataset
- custom metrics eval on eval_dataset
- saving of the final (last, best, ...) model version, together with the tokenizer

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
