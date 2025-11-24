# update as of 11-2025:
# 1. starts with loading basic modules and profiles
module load profile/base
module load cintools/1.0
module load anaconda3/2023.09-0
module load profile/deeplrn
# 2. loads core modules for fine-tuning, excluded aux modules since they are loaded automatically
module load gcc/12.2.0 cuda/12.2 nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22 cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.1 openmpi/4.1.6--gcc--12.2.0-cuda-12.2
