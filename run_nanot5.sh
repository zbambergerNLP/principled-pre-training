#!/bin/bash

# Example usage:
# sbatch run_pre_training_job.sh

#SBATCH --job-name="nanot5_pre_training_job"

#SBATCH -N 1                         # number of minimum nodes
#SBATCH --gres=gpu_cluster:6                 # Request n gpus
#SBATCH -c 100                       # number of cpus per task and per node

#SBATCH -p nlp
#SBATCH -A nlp

#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt       # stderr goes here
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

nvidia-smi

# Use independent adaptation of NanoT5
#accelerate launch main.py

# Use NanoT5 base code
cd nanoT5 || exit

python -m nanoT5.main \
    optim.name=adamwscale \
    optim.lr_scheduler=cosine